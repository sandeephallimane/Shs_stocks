import requests
import feedparser
from datetime import datetime
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import os
import time
import base64
import random
import glob


GAS_URL = os.getenv("GAS")
GEMINI_API_KEY = os.getenv("AK")
MUSIC_FOLDER = "Music"   # 👈 Folder in your repo where music files are stored

genai.configure(api_key=GEMINI_API_KEY)

def request_geminis(a):
        query = (
    "Fully read and analyze each news item in the provided text. Summarize the news in a neat bullet format.\n\n"
    "Instructions:\n"
    "- Exclude any news item that lacks logic or sufficient information.\n"
    "- Do not repeat any news items.\n"
    "- Exclude all film, entertainment, and sports news.\n"
    "- Organize the summary into the following sections: India, Global, State, Business, Economy, Science, Tech, and Other.\n"
    "- Start your response with: \"Summary of news item today:\"\n"
    "- At the end, include a \"Did You Know?\" section with two randomly chosen terms from the text. Provide a simple explanation for each.\n\n"
    f"Text:\n{a}"
        )

        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content(query)
        return resp.text.strip()
    
def fetch_rss_feeds(urls):
    all_entries = []
    for url in set(urls): 
        feed = feedparser.parse(url)
        for entry in feed.entries:
            summary = entry.get("summary") or entry.get("description") or "No summary available"
            all_entries.append({
                "title": entry.get("title", "No title"),
                "link": entry.get("link", "#"),
                "summary": summary,
            })
    return all_entries


def retry_request(func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                print(f"⚠️ Error: {e} — retrying in {delay} sec...")
                time.sleep(delay)
            else:
                raise


def get_podcast_script(news_text):
    def request_gemini():
        query = (
            "Read and analyze the provided news text, then create a podcast script "
            "for a single reader to read aloud using text-to-speech.\n\n"
            "Instructions:\n"
            "- Tone: Friendly, witty, conversational, yet informative.\n"
            "- Begin with a warm welcome and mention today's date.\n"
            "- smooth-flowing between sections.\n"
            "- Use natural transitions, no bullet lists.\n"
            "- Use '.........' to mark short pauses for pacing.\n"
            "- Highlight different sections.\n"
            "- Remove repeated news content and content that lacks full information.\n"  
            "- keep numbes in text format.\n"  
            "- Output ONLY the spoken script, no labels, commentary, or formatting.\n\n"
            f"News text to base the script on:\n{news_text}"
        )

        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content(query)
        return resp.text.strip()

    return retry_request(request_gemini)


def generate_podcast_audio(script_text, filename, music_folder=MUSIC_FOLDER):
    def tts_job():
        max_len = 4500  
        chunks = [script_text[i:i + max_len] for i in range(0, len(script_text), max_len)]

        final_audio = AudioSegment.silent(duration=0)
        for idx, chunk in enumerate(chunks):
            temp_file = f"chunk_{idx}.mp3"
            gTTS(text=chunk, lang="en-IN", tld="co.in", slow=False).save(temp_file)
            final_audio += AudioSegment.from_mp3(temp_file)
            os.remove(temp_file)

        music_files = glob.glob(os.path.join(music_folder, "*.mp3"))
        if music_files:
            bg_file = random.choice(music_files)
            print(f"🎶 Using background track: {bg_file}")

            bg_music = AudioSegment.from_mp3(bg_file)
            bg_music = bg_music - 18  

            if len(bg_music) < len(final_audio):
                repeat_count = (len(final_audio) // len(bg_music)) + 1
                bg_music = bg_music * repeat_count

            bg_music = bg_music[:len(final_audio)]
            final_audio = final_audio.overlay(bg_music)

        final_audio.export(filename, format="mp3")

    retry_request(tts_job)


if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing. Set it in your environment variables.")

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

entries = fetch_rss_feeds(rss_urls)
entries_text = "\n".join(
    f"{entry.get('title', '')}\n{entry.get('summary', '')}"
    for entry in entries
    if entry.get("summary") and entry["summary"].strip().lower() != "no summary available"
)

a = request_geminis(entries_text)
podcast_script = get_podcast_script(a)
if not podcast_script:
    raise ValueError("❌ Gemini returned an empty podcast script")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

transcript_file = f"podcast_transcript_{timestamp}.txt"
with open(transcript_file, "w", encoding="utf-8") as f:
    f.write(podcast_script)
print(f"✅ Transcript saved as {transcript_file}")

podcast_file = f"podcast_{timestamp}.mp3"
generate_podcast_audio(podcast_script, podcast_file, music_folder=MUSIC_FOLDER)
print(f"🎙 Podcast saved as {podcast_file}")

with open(podcast_file, "rb") as mp3_f:
    b64_audio = base64.b64encode(mp3_f.read()).decode("utf-8")

try:
    response = requests.post(
        GAS_URL,
        json={
            "type": "PODCAST",
            "filename": podcast_file,
            "content": b64_audio,
            "RN": "General News"
        }
    )
    response.raise_for_status()
    print("📤 GAS Upload Success:", response.text)
except Exception as e:
    print("❌ Failed to upload to GAS:", e)
            
