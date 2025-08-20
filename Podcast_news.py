import requests
from datetime import datetime
from gtts import gTTS
import os
import time

GAS_URL = "https://script.google.com/macros/s/AKfycbx9ynSOKtlNW1fTg_ZBMvSrPvwNI6X09UEVw-zIfG344biDkIb7XVEepQrCNFw7grg/exec"  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
entries_text = "..."  # Replace with your news text

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

        headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": query}]}],
            "generationConfig": {"temperature": 0.8, "topP": 0.9, "maxOutputTokens": 4096}
        }
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
            headers=headers,
            json=data,
            timeout=60
        )
        resp.raise_for_status()
        output = resp.json()
        return output["candidates"][0]["content"]["parts"][0]["text"]

    return retry_request(request_gemini)


def generate_podcast_audio(script_text, filename):
    def tts_job():
        tts = gTTS(text=script_text, lang="en", slow=False)
        tts.save(filename)
    retry_request(tts_job)


if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing. Set it in your GitHub Actions secrets.")


podcast_script = get_podcast_script(entries_text)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
transcript_file = f"podcast_transcript_{timestamp}.txt"
with open(transcript_file, "w", encoding="utf-8") as f:
    f.write(podcast_script)
print(f"✅ Transcript saved as {transcript_file}")

podcast_file = f"podcast_{timestamp}.mp3"
generate_podcast_audio(podcast_script, podcast_file)
print(f"🎙 Podcast saved as {podcast_file}")


with open(podcast_file, "rb") as mp3_f:
    try:
        response = requests.post(
            GAS_URL,
            files={"podcast": mp3_f},
            data={"type": "PODCAST"}
        )
        response.raise_for_status()
        print("📤 GAS Upload Success:", response.text)
    except Exception as e:
        print("❌ Failed to upload to GAS:", e)
