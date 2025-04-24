from datetime import datetime
import google.generativeai as genai
import os
import re
import requests
today_date = datetime.now().strftime("%B %d, %Y")
ak= os.getenv('AK')
genai.configure(api_key=ak)
GAS_URL = os.getenv('GAS')
model = genai.GenerativeModel("models/gemini-2.0-flash")

prompt =  f"""
You are a financial analyst. Summarize top finance news in India for {today_date} with headlines and 2-3 sentence summaries for each.
Then list major stocks in focus for today along with a brief reason for their movement (e.g. earnings, news, announcements).

Structure your response like this:

Top Finance News:
1. Headline: ...
   Summary: ...

2. Headline: ...
   Summary: ...

Stocks in Focus:
1. Stock Name: Reason
2. Stock Name: Reason
"""




def generate_html_from_output(text_output):
    html = "<html><head><title>Finance News</title></head><body>"
    lines = text_output.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Section Headers
        if line.startswith("**Top Finance News**") or line.startswith("**Stocks in Focus**"):
            html += f"<h2>{line.replace('**', '')}</h2>"

        # Headline with number
        elif re.match(r'^\d+\.\s+\*\*Headline:', line):
            headline = line.split("**Headline:")[1].split("**")[0].strip()
            html += f"<h3>{headline}</h3>"

        # Summary
        elif line.startswith("Summary:"):
            summary = line.replace("Summary:", "").strip()
            html += f"<p>{summary}</p>"

        # Stocks in Focus
        elif re.match(r'^\d+\.\s+\*\*(.+?):\*\*', line):
            match = re.match(r'^\d+\.\s+\*\*(.+?):\*\*\s+(.*)', line)
            if match:
                stock_name = match.group(1).strip()
                reason = match.group(2).strip()
                html += f"<p><strong>{stock_name}:</strong> {reason}</p>"

        # General fallback (e.g., opening paragraph)
        elif line:
            html += f"<p>{line}</p>"

    html += "</body></html>"
    return html

response = model.generate_content(prompt)
html_output = generate_html_from_output(response.text)
print(html_output)
response = requests.post(GAS_URL, data={"html": html_output})
  print(response.text)
