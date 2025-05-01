import feedparser
import os
from datetime import datetime
import google.generativeai as genai
import re
import requests
from bs4 import BeautifulSoup
today_date = datetime.now().strftime("%B %d, %Y")
ak= os.getenv('AK')
genai.configure(api_key=ak)
GAS_URL = os.getenv('GAS')
rss_urls = [ 
    "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
    "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/economy.xml",
    "https://www.livemint.com/rss/markets.xml",  
    "https://cfo.economictimes.indiatimes.com/rss/topstories",  
    "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss"
]

def fetch_rss_feeds(urls):
    all_entries = []
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            all_entries.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.summary,
            })
    return all_entries

# Function to generate HTML output
def generate_html(entries):
    html_content = """
    <html>
    <head>
        <title>RSS Feed Aggregator</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .feed-item { margin-bottom: 20px; }
            .feed-item a { font-size: 18px; color: #2a7cf8; text-decoration: none; }
            .feed-item a:hover { text-decoration: underline; }
            .feed-item p { font-size: 14px; color: #555; }
        </style>
    </head>
    <body>
        <h1>Greetings,Latest News from Multiple RSS Feeds</h1>
    """
    
    for entry in entries:
        html_content += f"""
        <div class="feed-item">
            <a href="{entry['link']}">{entry['title']}</a>
            <p>{entry['summary']}</p>
        </div>
        """
    
    html_content += "</body></html>"
    
    return html_content

entries = fetch_rss_feeds(rss_urls)
html_output = generate_html(entries)

print(html_output)
response = requests.post(GAS_URL, data={"html": html_output})
print(response.text)

#rss_url = "https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml"
#feed = feedparser.parse(rss_url)

#links = []
#for entry in feed.entries:
   # links.append(entry.link)
#print(links)
