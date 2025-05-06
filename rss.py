import feedparser
import os
from datetime import datetime
import google.generativeai as genai
import re
import htmlmin
import requests
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import guess_lexer, TextLexer
from pygments.formatters import HtmlFormatter
import markdown

def is_probable_markdown(text):
    return bool(re.search(r"(^# |\*\*|`|^- |\n\d+\. )", text, re.M))

def is_probable_code(text):
    lines = text.strip().splitlines()
    return len(lines) > 1 and all(re.match(r'\s{2,}', line) or re.match(r'\w+\s*=', line) for line in lines[:3])

def convert_to_html(text: str) -> str:
    content_html = ""

    if is_probable_markdown(text):
        content_html = markdown.markdown(text)
    elif is_probable_code(text):
        try:
            lexer = guess_lexer(text)
        except Exception:
            lexer = TextLexer()
        formatter = HtmlFormatter(noclasses=True, style="friendly")
        content_html = highlight(text, lexer, formatter)
    else:
        content_html = f"<pre>{text}</pre>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Dynamic Content</title>
  <style>
    body {{
      font-family: 'Segoe UI', sans-serif;
      background-color: #f4f4f4;
      margin: 0;
      padding: 2rem;
    }}
    .container {{
      max-width: 900px;
      margin: auto;
      background: #fff;
      padding: 2rem;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      overflow-wrap: break-word;
    }}
  </style>
</head>
<body>
  <div class="container">
    {content_html}
  </div>
</body>
</html>
"""
    return html
    
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

def gemini_response_to_html(gemini_response: str) -> str:
    sections = re.split(r"\n\s*\*\*(.+?)\*\*:\s*\n", gemini_response.strip())
    
    # Handle first part (intro, if any)
    html_sections = []
    intro = sections[0]
    if intro:
        html_sections.append(f"<p>{intro.strip()}</p>")
    
    # Process each section with a header and its items
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        bullet_text = sections[i+1]

        # Parse bullet points in this section
        bullets = re.findall(r"\*\s+\*\*(.+?)\*\*:(.+?)(?=\n\*|\Z)", bullet_text.strip(), flags=re.DOTALL)

        bullet_items = "".join(
            f"<li><b>{title.strip()}:</b> {desc.strip()}</li>" for title, desc in bullets
        )

        section_html = f"""
        <h2>{section_title}</h2>
        <ul>
            {bullet_items}
        </ul>
        """
        html_sections.append(section_html)

    # HTML email template
    html = f"""
    <html>
    <head>
      <style>
        body {{
          font-family: Arial, sans-serif;
          background-color: #f4f4f4;
          padding: 20px;
          color: #333;
        }}
        .container {{
          background-color: #fff;
          padding: 30px;
          border-radius: 10px;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h2 {{
          color: #0066cc;
          border-bottom: 2px solid #e0e0e0;
          padding-bottom: 5px;
        }}
        ul {{
          padding-left: 20px;
        }}
        li {{
          margin-bottom: 10px;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        {''.join(html_sections)}
      </div>
    </body>
    </html>
    """
    return html



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
ht = convert_to_html(j.text)
text = re.sub(r"\*\s+\*\*", r"\n• ", j.text)
text = text.replace("**", " ")
response = requests.post(GAS_URL, data={"html": ht})
print(response.text)

#rss_url = "https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml"
#feed = feedparser.parse(rss_url)

#links = []
#for entry in feed.entries:
   # links.append(entry.link)
#print(links)
