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

def generate_html(entries):
    html_content = """
    <html>
    <head>
        <title>RSS Feed Aggregator</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

            body {
                font-family: 'Inter', sans-serif;
                background: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 0;
                transition: background 0.3s, color 0.3s;
            }

            :root {
                --bg: #f0f2f5;
                --text: #2e3a59;
                --card-bg: #ffffff;
                --link: #3182ce;
            }

            body.dark {
                --bg: #1a202c;
                --text: #e2e8f0;
                --card-bg: #2d3748;
                --link: #63b3ed;
            }

            .container {
                max-width: 900px;
                margin: 50px auto;
                background: var(--card-bg);
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
                transition: background 0.3s ease-in-out;
            }

            h1 {
                text-align: center;
                color: var(--text);
                margin-bottom: 40px;
                font-weight: 600;
            }

            .feed-item {
                margin-bottom: 25px;
                padding-bottom: 20px;
                border-bottom: 1px solid #ccc;
                animation: fadeIn 0.5s ease forwards;
                opacity: 0;
            }

            .feed-item:last-child {
                border-bottom: none;
            }

            .feed-item a {
                font-size: 20px;
                color: var(--link);
                text-decoration: none;
                font-weight: 600;
                display: inline-block;
                margin-bottom: 10px;
                transition: color 0.2s ease;
            }

            .feed-item a:hover {
                color: darken(var(--link), 10%);
            }

            .feed-item p {
                font-size: 16px;
                display: none;
                margin: 0;
                color: var(--text);
            }

            .toggle-button {
                cursor: pointer;
                font-size: 14px;
                color: #999;
                margin-top: 5px;
                display: inline-block;
            }

            .toggle-dark {
                position: fixed;
                top: 20px;
                right: 20px;
                background: none;
                border: 1px solid #ccc;
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                background-color: var(--card-bg);
                color: var(--text);
                transition: all 0.3s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <button class="toggle-dark" onclick="toggleDarkMode()">Toggle Mode</button>
        <div class="container">
            <h1>Curated News from Your Favorite Sources</h1>
    """

    for idx, entry in enumerate(entries):
        html_content += f"""
            <div class="feed-item" style="animation-delay: {idx * 0.1}s">
                <a href="{entry['link']}" target="_blank">{entry['title']}</a>
                <div class="toggle-button" onclick="toggleSummary('summary-{idx}')">Show Summary</div>
                <p id="summary-{idx}">{entry['summary']}</p>
            </div>
        """

    html_content += """
        </div>
        <script>
            function toggleSummary(id) {
                var el = document.getElementById(id);
                if (el.style.display === 'none' || el.style.display === '') {
                    el.style.display = 'block';
                } else {
                    el.style.display = 'none';
                }
            }

            function toggleDarkMode() {
                document.body.classList.toggle('dark');
            }

            // Trigger fade-in animation
            window.addEventListener('DOMContentLoaded', () => {
                const items = document.querySelectorAll('.feed-item');
                items.forEach((el, i) => {
                    setTimeout(() => {
                        el.style.opacity = 1;
                    }, i * 100);
                });
            });
        </script>
    </body>
    </html>
    """

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
