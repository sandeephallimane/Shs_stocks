import os
import time
import glob
import base64
import random
import asyncio
import requests
import feedparser
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pydub import AudioSegment
import google.generativeai as genai
import edge_tts


IST = ZoneInfo("Asia/Kolkata")
NOW_IST = datetime.now(IST)
TODAY_DATE = NOW_IST.strftime("%Y-%m-%d - %A")


GAS_URL = os.getenv("GAS")
GEMINI_API_KEY = os.getenv("AK")


MUSIC_FOLDER = "Music"  # Background music folder
DEFAULT_VOICE = os.getenv("VOICE", "en-IN-NeerjaNeural")  # Voice: en-IN-NeerjaNeural / en-IN-PrabhatNeural


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing. Set it in your environment variables.")
genai.configure(api_key=GEMINI_API_KEY)



def fetch_rss_feeds(urls):
    all_entries = []
    for url in set(urls):
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                summary = entry.get("summary") or entry.get("description") or "No summary available"
                all_entries.append({
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", "#"),
                    "summary": summary,
                })
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch {url}: {e}")
    return all_entries


def retry_request(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"⚠️ Error: {e} — retrying in {delay} sec...")
                time.sleep(delay)
            else:
                logging.error(f"❌ Failed after {retries} attempts: {e}")
                raise


def request_geminis(news_text):
    query = (
        "Fully analyze the provided text and summarize the news clearly.\n\n"
        "Instructions:\n"
        "- Exclude irrelevant, duplicate, or incomplete stories.\n"
        "- Exclude film, entertainment, and sports.\n"
        "- Organize into sections: India, Global, Business & Economy, Science & Tech, Other.\n"
        "- Output structured text for further use.\n\n"
        f"Text:\n{news_text}"
    )
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    resp = model.generate_content(query)
    return resp.text.strip()


def get_podcast_script(news_summary):
    def request_script():
        query = (
            "You are a professional news anchor creating a podcast script from the provided summarized news.\n\n"
            "Instructions:\n"
            f"- Begin with a professional, warm welcome and mention today's date: {TODAY_DATE}.\n"
            "- Organize into sections: India, Global, Business & Economy, Science & Tech, Other.\n"
            "- Within each section:\n"
            "   * Present each news item as a short, clear spoken narration.\n"
            "   * Insert '.........' after each individual news item for a short pause.\n"
            "- At the end of each section, insert '...............' to indicate a longer pause before the next section.\n"
            "- Use natural transitions, no bullet lists.\n"
            "- Keep tone authoritative, professional, yet listener-friendly.\n"
            "- Remove incomplete or repeated news.\n"
            "- End with a professional closing line thanking listeners.\n"
            "- Output ONLY the spoken script.\n\n"
            f"Summarized news:\n{news_summary}"
        )
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content(query)
        return resp.text.strip()

    return retry_request(request_script)


async def tts_edge(script_text, filename, voice=DEFAULT_VOICE):
    communicate = edge_tts.Communicate(script_text, voice=voice, rate="+0%")
    await communicate.save(filename)


def generate_podcast_audio(script_text, filename, music_folder=MUSIC_FOLDER, voice=DEFAULT_VOICE):
    def tts_job():
        max_len = 4000
        chunks = [script_text[i:i + max_len] for i in range(0, len(script_text), max_len)]
        final_audio = AudioSegment.silent(duration=0)

        for idx, chunk in enumerate(chunks):
            temp_file = f"chunk_{idx}.mp3"
            asyncio.run(tts_edge(chunk, temp_file, voice))
            final_audio += AudioSegment.from_mp3(temp_file)
            os.remove(temp_file)

        music_files = glob.glob(os.path.join(music_folder, "*.mp3"))
        if music_files:
            bg_file = random.choice(music_files)
            logging.info(f"🎶 Using background track: {bg_file}")
            bg_music = AudioSegment.from_mp3(bg_file) - 18
            if len(bg_music) < len(final_audio):
                bg_music *= (len(final_audio) // len(bg_music)) + 1
            bg_music = bg_music[:len(final_audio)]
            final_audio = final_audio.overlay(bg_music)

        final_audio.export(filename, format="mp3")

    retry_request(tts_job)


def upload_to_gas(file_path, file_type="PODCAST", rn="General News"):
    """Upload generated podcast to Google Apps Script."""
    with open(file_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(
            GAS_URL,
            json={
                "type": file_type,
                "filename": os.path.basename(file_path),
                "content": b64_audio,
                "RN": rn
            }
        )
        response.raise_for_status()
        logging.info(f"📤 GAS Upload Success: {response.text}")
    except Exception as e:
        logging.error(f"❌ Failed to upload to GAS: {e}")


if __name__ == "__main__":
    rss_urls = [
        "https://indianexpress.com/feed/",
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://timesofindia.indiatimes.com/rssfeedmostrecent.cms",
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://www.thehindu.com/news/international/feeder/default.rss",
        "https://www.news18.com/commonfeeds/v1/eng/rss/india.xml",
        "https://www.news18.com/commonfeeds/v1/eng/rss/world.xml",
        "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/india.xml",
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://www.dnaindia.com/feeds/india.xml",
        "https://www.firstpost.com/commonfeeds/v1/mfp/rss/india.xml",
        "https://www.firstpost.com/commonfeeds/v1/mfp/rss/world.xml",
        "https://zeenews.india.com/rss/world-news.xml",
        "https://zeenews.india.com/rss/india-national-news.xml",
        "https://zeenews.india.com/rss/india-news.xml"
    ]

    logging.info("📥 Fetching RSS feeds...")
    entries = fetch_rss_feeds(rss_urls)
    entries_text = "\n".join(
        f"{entry.get('title', '')}\n{entry.get('summary', '')}"
        for entry in entries
        if entry.get("summary") and entry["summary"].strip().lower() != "no summary available"
    )

    logging.info("📝 Summarizing news...")
    summary = request_geminis(entries_text)

    logging.info("🎙 Creating podcast script...")
    podcast_script = get_podcast_script(summary)
    if not podcast_script:
        raise ValueError("❌ Gemini returned an empty podcast script")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    transcript_file = f"podcast_transcript_{timestamp}.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(podcast_script)
    logging.info(f"✅ Transcript saved: {transcript_file}")

    podcast_file = f"podcast_{timestamp}.mp3"
    generate_podcast_audio(podcast_script, podcast_file, music_folder=MUSIC_FOLDER, voice=DEFAULT_VOICE)
    logging.info(f"🎧 Podcast saved: {podcast_file}")

    
    upload_to_gas(podcast_file)
          
