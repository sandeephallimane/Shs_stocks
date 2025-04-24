from datetime import datetime
import google.generativeai as genai
import os
today_date = datetime.now().strftime("%B %d, %Y")
ak= os.getenv('AK')
genai.configure(api_key=ak)

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

response = model.generate_content(prompt)
print(response.text)
