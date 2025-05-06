import feedparser
import os
from datetime import datetime
import google.generativeai as genai
import re
import htmlmin
import requests
from bs4 import BeautifulSoup
today_date = datetime.now().strftime("%B %d, %Y")
ak= os.getenv('AK')
genai.configure(api_key=ak)
GAS_URL = os.getenv('GAS')
model = genai.GenerativeModel("models/gemini-2.0-flash")  
rss_urls = [ 
    "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
    "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/economy.xml",
    "https://www.livemint.com/rss/markets.xml",  
    "https://cfo.economictimes.indiatimes.com/rss/topstories",  
    "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss",
]

def fetch_rss_feeds(urls):
    all_entries = []
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            summary = entry.get('summary') or entry.get('description') or "No summary available"
            all_entries.append({
                "title": entry.get('title', 'No title'),
                "link": entry.get('link', '#'),
                "summary": summary,
            })
    return all_entries


def generate_html(entries):
    html_content = """
    <html>
    <body style="margin:0; padding:20px; background-color:#f4f6f9; font-family: Arial, sans-serif;">
        <table align="center" cellpadding="0" cellspacing="0" width="100%" style="max-width:600px; background-color:#ffffff; padding:20px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
            <!-- Title -->
            <tr>
                <td align="center" style="font-size:26px; font-weight:bold; color:#333333; padding-bottom:30px;">
                    Curated News Just for You
                </td>
            </tr>
    """

    # Loop through the entries and create each feed item
    for entry in entries:
        html_content += f"""
            <tr>
                <td style="padding:20px 0; border-top:1px solid #f0f0f0; border-bottom:1px solid #f0f0f0;">
                    <a href="{entry['link']}" style="text-decoration:none; font-size:20px; color:#1a73e8; font-weight:bold; display:block; padding:10px 0; transition:color 0.3s ease;">
                        {entry['title']}
                    </a>
                    <p style="margin:10px 0 20px; font-size:16px; color:#555555; line-height:1.6; font-weight:300;">
                        {entry['summary']}
                    </p>
                    <a href="{entry['link']}" style="text-decoration:none; font-size:14px; color:#1a73e8; font-weight:bold; display:block; text-align:center; background-color:#e0f7fa; padding:10px 0; border-radius:5px; transition:background-color 0.3s ease;">Read More</a>
                </td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    return html_content
          

entries = fetch_rss_feeds(rss_urls)
print(entries)
query = "Read and summarize below news items in neat bullet format\n" + "\n".join(
    entry['title'] for entry in entries
)

j=model.generate_content(query)
print(j.text)
html_output = generate_html(entries)
minified_html = htmlmin.minify(
        html_output,
        remove_comments=True,
        remove_empty_space=True,
        reduce_boolean_attributes=True,
        remove_optional_attribute_quotes=False
    )

response = requests.post(GAS_URL, data={"html": minified_html})
print(response.text)

#rss_url = "https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml"
#feed = feedparser.parse(rss_url)

#links = []
#for entry in feed.entries:
   # links.append(entry.link)
#print(links)
