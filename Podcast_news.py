import requests
import feedparser
from datetime import datetime
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import os
import time
import base64

# ----------------- CONFIG -----------------
GAS_URL = "https://script.google.com/macros/s/AKfycbw2F-uOtzNJrPNkRCg4MzRp76Jg_khb0fRqFFlW9k92kUsxbDlAuBqZP8yncPoDrefy/exec"
GEMINI_API_KEY = os.getenv("AK")
genai.configure(api_key=GEMINI_API_KEY)

# ----------------- FUNCTIONS -----------------
def fetch_rss_feeds(urls):
    """Fetch news entries from a list of RSS URLs."""
    all_entries = []
    for url in set(urls):  # ✅ Deduplicate URLs
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
    """Retry wrapper for unstable API calls."""
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
    """Generate a podcast script using Gemini API."""
    def request_gemini():
        query = (
            "Read and analyze the provided news text and create a fun, engaging podcast script "
            "as a conversation between two hosts, Host 1 and Host 2.\n\n"
            "Instructions:\n"
            "- Use a friendly, witty tone while staying informative.\n"
            "- Start with a short welcome and today's date.\n"
            "- Have hosts naturally alternate lines — Host 1 then Host 2, back and forth.\n"
            "- Cover India, Global, State, Business, Economy, Science, Tech, and Other news.\n"
            "- No film, entertainment, or sports news.\n"
            "- End with a 'Did You Know?' section with two interesting facts.\n"
            "- Output only the spoken lines, prefixing each with 'Host 1:' or 'Host 2:'.\n\n"
            f"Text:\n{news_text}"
        )
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        resp = model.generate_content(query)
        return resp.text.strip()

    return retry_request(request_gemini)


def generate_podcast_audio(script_text, filename):
    """Generate TTS audio from podcast script, splitting if too long."""
    def tts_job():
        max_len = 4500  # gTTS text limit
        chunks = [script_text[i:i + max_len] for i in range(0, len(script_text), max_len)]

        final_audio = AudioSegment.silent(duration=0)
        for idx, chunk in enumerate(chunks):
            temp_file = f"chunk_{idx}.mp3"
            gTTS(text=chunk, lang="en", slow=False).save(temp_file)
            final_audio += AudioSegment.from_mp3(temp_file)
            os.remove(temp_file)

        final_audio.export(filename, format="mp3")

    retry_request(tts_job)


# ----------------- MAIN LOGIC -----------------
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

# Fetch and prepare news
entries = fetch_rss_feeds(rss_urls)
entries_text = "\n".join(
    f"{entry.get('title', '')}\n{entry.get('summary', '')}"
    for entry in entries
    if entry.get("summary") and entry["summary"].strip().lower() != "no summary available"
)

# Generate podcast script
podcast_script = get_podcast_script(entries_text)
if not podcast_script:
    raise ValueError("❌ Gemini returned an empty podcast script")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save transcript
transcript_file = f"podcast_transcript_{timestamp}.txt"
with open(transcript_file, "w", encoding="utf-8") as f:
    f.write(podcast_script)
print(f"✅ Transcript saved as {transcript_file}")

# Generate podcast MP3
podcast_file = f"podcast_{timestamp}.mp3"
generate_podcast_audio(podcast_script, podcast_file)
print(f"🎙 Podcast saved as {podcast_file}")

with open(podcast_file, "rb") as mp3_f:
    b64_audio = base64.b64encode(mp3_f.read()).decode("utf-8")

try:
    response = requests.post(
        GAS_URL,
        json={  # 👈 JSON, not files/data
            "type": "PODCAST",
            "filename": podcast_file,
            "content": b64_audio
        }
    )
    response.raise_for_status()
    print("📤 GAS Upload Success:", response.text)
except Exception as e:
    print("❌ Failed to upload to GAS:", e)
