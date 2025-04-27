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
    "https://www.moneycontrol.com/rss/latestnews.xml",  # Moneycontrol Latest News
    "https://www.livemint.com/rss/markets.xml",  # Economic Times Latest News
    "https://cfo.economictimes.indiatimes.com/rss/topstories",  # NDTV Profit
    "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss",  # Business Standard Markets
    "https://www.moneycontrol.com/rss/stocksinfocus.xml",  # Moneycontrol Stocks in Focus
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

rss_url = "https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml"
feed = feedparser.parse(rss_url)

links = []
for entry in feed.entries:
    links.append(entry.link)

data_list = []

for link in links:
    try:
        response = requests.get(link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')

            # Extract important fields
            company_name = soup.find('in-capmkt:NameOfTheCompany').text if soup.find('in-capmkt:NameOfTheCompany') else ''
            symbol = soup.find('in-capmkt:NSESymbol').text if soup.find('in-capmkt:NSESymbol') else ''
            isin = soup.find('in-capmkt:ISIN').text if soup.find('in-capmkt:ISIN') else ''
            name_of_person = soup.find('in-capmkt:NameOfThePerson').text if soup.find('in-capmkt:NameOfThePerson') else ''
            transaction_type = soup.find('in-capmkt:SecuritiesAcquiredOrDisposedTransactionType').text if soup.find('in-capmkt:SecuritiesAcquiredOrDisposedTransactionType') else ''
            no_of_securities = soup.find('in-capmkt:SecuritiesAcquiredOrDisposedNumberOfSecurity').text if soup.find('in-capmkt:SecuritiesAcquiredOrDisposedNumberOfSecurity') else ''
            value_of_securities = soup.find('in-capmkt:SecuritiesAcquiredOrDisposedValueOfSecurity').text if soup.find('in-capmkt:SecuritiesAcquiredOrDisposedValueOfSecurity') else ''
            mode_of_acquisition = soup.find('in-capmkt:ModeOfAcquisitionOrDisposal').text if soup.find('in-capmkt:ModeOfAcquisitionOrDisposal') else ''
            date_of_acquisition = soup.find('in-capmkt:DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate').text if soup.find('in-capmkt:DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate') else ''

            data_list.append({
                "Company Name": company_name,
                "Symbol": symbol,
                "ISIN": isin,
                "Person Name": name_of_person,
                "Transaction Type": transaction_type,
                "No of Securities": no_of_securities,
                "Value of Securities": value_of_securities,
                "Mode of Acquisition/Disposal": mode_of_acquisition,
                "Date of Acquisition/Sale": date_of_acquisition,
                "Link": link
            })
    except Exception as e:
        print(f"Failed to fetch {link}: {e}")

html_table = """
<table border="1" cellpadding="5" cellspacing="0">
<thead>
<tr>
<th>Company Name</th>
<th>Symbol</th>
<th>ISIN</th>
<th>Person Name</th>
<th>Transaction Type</th>
<th>No of Securities</th>
<th>Value of Securities</th>
<th>Mode of Acquisition/Disposal</th>
<th>Date of Acquisition/Sale</th>
<th>Link</th>
</tr>
</thead>
<tbody>
"""

for row in data_list:
    html_table += "<tr>"
    for key, value in row.items():
        if key == "Link":
            html_table += f"<td><a href='{value}' target='_blank'>View XML</a></td>"
        else:
            html_table += f"<td>{value}</td>"
    html_table += "</tr>"

html_table += """
</tbody>
</table>
"""

print(html_table)
response1 = requests.post(GAS_URL, data={"html": html_table})
print(response1.text)
