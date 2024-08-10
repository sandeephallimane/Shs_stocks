import pandas as pd
import yfinance as yf
from pmdarima.arima import auto_arima
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import kurtosis  # Import kurtosis function from scipy.stats
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
from jinja2 import Environment, FileSystemLoader
#import pdfkit
from weasyprint import HTML, CSS
import os
import google.generativeai as genai

ticker_symbols=(os.getenv('TS')).split(',')
print(ticker_symbols)
ak= os.getenv('AK')
se=os.getenv('SE')
re=(os.getenv('RE')).split(',')
print(re)
pwd = os.getenv('PASSWORD')
if pwd is None :
    raise ValueError("Password not found in environment variables")
if ak is None :
    raise ValueError("API Key not found in environment variables")
if se is None :
    raise ValueError("Sending Email not found in environment variables")
if re is None :
    raise ValueError("Receiver Email not found in environment variables")
genai.configure(api_key=ak)
current_time_ist = (datetime.now() + timedelta(hours=5, minutes=30, seconds=0)).strftime("%Y-%m-%d %H:%M:%S") 
current_date = datetime.now().date()
five_years_ago = current_date - timedelta(days=5 * 365)
ms= 'Stock Forecast Results:'+current_time_ist
# Format the dates as strings
start_date = five_years_ago.strftime('%Y-%m-%d')
end_date = current_date.strftime('%Y-%m-%d')
# Function to fetch and process data for each ticker symbol
def calculate_ema(values, window):
    return values.ewm(span=window, adjust=False).mean()
current_date = datetime.now().date()
five_years_ago = current_date - timedelta(days=5 * 365)
def calculate_stochastic_oscillator(df):
    n=14
    m=3
    df['Lowest Low'] = df['Close'].rolling(window=n).min()
    df['Highest High'] = df['Close'].rolling(window=n).max()
    df['Stochastic_K'] = ((df['Close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low'])) * 100
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=m).mean()
    df.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)
    return "Buy" if df['Stochastic_K'].iloc[-1] > df['Stochastic_D'].iloc[-1] and 50 < df['Stochastic_K'].iloc[-1] > 20else "Sell"

def cal_GC(data):
  short_ema = calculate_ema(data, 50)
  long_ema = calculate_ema(data, 200)
  return "Buy" if short_ema.iloc[-1] > long_ema.iloc[-1] else "Sell"
def cal_MACD(data):
  short_ema = calculate_ema(data, 12)
  long_ema = calculate_ema(data, 26)
  macd_line = short_ema - long_ema
  signal_line = calculate_ema(macd_line, 9)
  merged_df = pd.merge(macd_line, signal_line, left_index=True, right_index=True, how='inner',suffixes=('_MACD', '_SL'))
  merged_df = merged_df.dropna()
  return "Buy" if merged_df.iloc[-1,0]>merged_df.iloc[-1,1] else "Sell"
def calculate_rsi(data, window=14):
  delta = data.diff().tail(252)
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)
  avg_gain = gain.ewm(span=window, min_periods=window).mean()
  avg_loss = loss.ewm(span=window, min_periods=window).mean()
  rs = avg_gain / avg_loss
  rsi = 100 - (100 / (1 + rs))
  return (rsi.tail(1).iloc[0]).round(2)
    
def generate_pdf(html_content,footer_html):
    try:
        options = {
            'page-size': 'A4',
            'margin-top': '1cm',
            'margin-right': '1cm',
            'margin-bottom': '1.5cm',
            'margin-left': '1cm',
            'footer-html': footer_html,  # Path to footer HTML file
        }
        
       # pdfkit.from_string(html_content, 'Arima_forecast_summary.pdf', options=options)
       # print("PDF created successfully: Arima_forecast_summary.pdf")
    except Exception as e:
        print(f"Error creating PDF: {e}")
def fndmntl(ticker):
    stock = yf.Ticker(ticker)
    return stock   
    
def forecast_stock_returns(ticker_symbol):
    print(ticker_symbol)
    try:
      stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
      stock_data.dropna(inplace=True)
      if len(stock_data)>300:
        # Calculate daily returns
        stock_data['Returns'] =  np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
        stock_data['Diff'] =  stock_data['Adj Close'].diff()
        stock_data.dropna(inplace=True)
        stock_data['52 Week High'] = stock_data['Adj Close'].rolling(window=252).max()
        stock_data['52 Week Low'] = stock_data['Adj Close'].rolling(window=252).min()
        last_date = stock_data.index[-1].date()
        stock_data_52_high = stock_data['52 Week High'].iloc[-1].round(2)
        stock_data_52_low = stock_data['52 Week Low'].iloc[-1].round(2)
        cv = stock_data['Returns'].std() / stock_data['Returns'].mean()
        kurtosis_val =kurtosis(stock_data['Returns'])
        rsi = calculate_rsi(stock_data['Adj Close'].tail(252), window=14) 
        current_cmp = stock_data['Adj Close'].iloc[-1].round(2)
        yearly_returns = (pd.Series.mean(stock_data['Returns']) * 25000).round(2)
        TI=[]
        TI.append('Buy' if 45 > (calculate_rsi(stock_data['Close'].tail(700))) > 30 else 'Sell')
        TI.append(cal_MACD((stock_data['Close']).tail(700)))
        TI.append(cal_GC((stock_data['Close']).tail(700)))
        TI.append(calculate_stochastic_oscillator((stock_data).tail(700)))
        if yearly_returns>12 and current_cmp>20 and current_cmp < (stock_data_52_high*0.9):
            z_scores = np.abs(stock_data['Returns'] - stock_data['Returns'].mean()) / stock_data['Returns'].std()
            stock_data = stock_data[(z_scores < 3)]  
            z_scores1 = np.abs(stock_data['Diff'] - stock_data['Diff'].mean()) / stock_data['Diff'].std()
            stock_data = stock_data[(z_scores1 < 3)]
            series = stock_data['Returns']
            series1 = stock_data['Diff']
            model11 = auto_arima(series, seasonal=False, trace=False,start_P=0,start_D=0,start_Q=0,max_P=5,max_D=5,max_Q=5)
            model12 = auto_arima(series, seasonal=True, trace=False,start_P=0,start_D=0,start_Q=0,max_P=5,max_D=5,max_Q=5)
            model21 = auto_arima(series1, seasonal=False, trace=False,start_P=0,start_D=0,start_Q=0,max_P=5,max_D=5,max_Q=5)
            model22 = auto_arima(series1, seasonal=True, trace=False,start_P=0,start_D=0,start_Q=0,max_P=5,max_D=5,max_Q=5)
            forecast_periods = 120
            forecast11 = model11.predict(n_periods=forecast_periods)
            forecast12 = model12.predict(n_periods=forecast_periods)
            forecast21 = model21.predict(n_periods=forecast_periods)
            forecast22 = model22.predict(n_periods=forecast_periods)
            v= [yearly_returns,current_cmp,stock_data_52_high,stock_data_52_low,rsi]
            res11= [current_cmp * np.cumprod(1 + np.array(forecast11[:i+1]))[-1] for i in range(len(forecast11))]
            res_p11= ["Return-Non Seasonal",np.average(res11).round(2),np.max(res11).round(2),np.min(res11).round(2),(((np.average(res11)-current_cmp)*100)/current_cmp).round(2)]
            res12= [current_cmp * np.cumprod(1 + np.array(forecast12[:i+1]))[-1] for i in range(len(forecast12))]
            res_p12= ["Return-Seasonal", np.average(res12).round(2),np.max(res12).round(2),np.min(res12).round(2),(((np.average(res12)-current_cmp)*100)/current_cmp).round(2)]
            res21 = [current_cmp]+ [current_cmp + sum(forecast21[:i+1]) for i in range(len(forecast21))]
            res_p21= ["Price Diff-Non Seasonal",np.average(res21).round(2),np.max(res21).round(2),np.min(res21).round(2),(((np.average(res21)-current_cmp)/current_cmp)*100).round(2)]
            res22 = [current_cmp]+ [current_cmp + sum(forecast22[:i+1]) for i in range(len(forecast22))]
            res_p22=["Price Diff Seasonal",np.average(res22).round(2),np.max(res22).round(2),np.min(res22).round(2),(((np.average(res22)-current_cmp)/current_cmp)*100).round(2)]
            k= [ticker_symbol,v,[res_p11,res_p12,res_p21,res_p22],"NA",TI]
            return k
        else:
            return "NA"
      else:
           return "NA"
    except Exception as e:
         print(f"Error fetching data for {ticker_symbol}: {str(e)}")
         return "NA"

forecasts = []
t =[]
for ticker_symbol in ticker_symbols:
    result = forecast_stock_returns(ticker_symbol)
    if result != "NA":
       if (((result[2])[0])[4])>5 and (((result[2])[2])[4])>5:
          a= fndmntl(result[0])
          query = "Read and summarize financial position/n"+ (((a.balance_sheet).iloc[:, :2]).dropna()).to_string() + "and "+(((a.financials).iloc[:, :2]).dropna()).to_string()+ "and "+(((a.financials).iloc[:, :2]).dropna()).to_string()
          model = genai.GenerativeModel("models/gemini-1.0-pro")  
          j=model.generate_content(query)
          result[3] = j.text
          forecasts.append(result)
    result= []

def generate_nested_html(single_row): 
    # Start the main row
    nested_html = "<tr>"
    nested_html += f"<td>{single_row[0]}</td>"
    
    nested_html += "<td>"
    nm =["Yearly Ret%","CMP","52 Wk High","52 Wk Low","RSI"]  
    nested_html += "<table border='1'>\n"
    # Iterate over the minimum length to avoid index errors
    for i, value in enumerate(single_row[1][:min(len(nm), len(single_row[1]))]):
        nested_html += "<tr>" 
        nested_html += f"<td>{nm[i]}</td>"
        nested_html += f"<td>{value}</td>"
        nested_html += "</tr>\n"
    nested_html += "</table>\n"
    nested_html += "</td>"
    nested_html += "<td colspan='5'>"
    nested_html += f"<table border='1'>\n"
    for i, sublist in enumerate(single_row[2]):
        nested_html += "<tr>"
        for j, row in enumerate(sublist):
            nested_html += f"<td>{row}</td>"   
        nested_html += "</tr>"           
    nested_html += "</table>\n"
    nested_html += "</td>"
    nested_html += "<td>"
    nm1 =["RSI","MACD","EMA","Stochastic Oscilator"]  
    nested_html += "<table border='1'>\n"
    # Iterate over the minimum length to avoid index errors
    for i, value in enumerate(single_row[4][:min(len(nm1), len(single_row[4]))]):
        nested_html += "<tr>" 
        nested_html += f"<td>{nm1[i]}</td>"
        nested_html += f"<td>{value}</td>"
        nested_html += "</tr>\n"
    nested_html += "</table>\n"
    nested_html += "</td>"
    nested_html += f"<td>{single_row[3]}</td>"
    nested_html += "</tr>\n"    
    return nested_html
    
email_body = """
<html>
<head>
  <style>
    @page {
            size: A4;
            margin: 1cm;
            border: 0.3px solid purple ; 
            border-radius: 5px;
            padding: 13px;
            @bottom-right {
                content: counter(page);
            }
        }
    table {
      font-family: 'Times New Roman', Times, serif;
      font-size: 5px;
      width: 100%;
      page-break-inside: avoid;
    }
    th, td {
      border: 1px solid #dddddd;
      text-align: center;
      padding: 8px;
      border-radius: 2.5px;
      color: #410044;
      border-collapse: separate;
      border-radius:2px;
      page-break-inside: avoid; /* Avoid breaking inside table cells */
    }
    .footer {
            text-align: center;
            margin-top: 20px;
        }
    th {
      background-color: #E8FF74;
    }
  </style>
</head>
<body>
<h1 style="text-align:center;color: #440000;"> <u> Weekly Arima Forecast Summary </u></h1>
   <table><tr>
    <th rowspan="2">Stock Name </th>
    <th rowspan="2">Details</th>
    <th colspan="5">Forecast Details</th>
    <th  rowspan="2">Technical Indicators</th>
    <th rowspan="2">Summary </th>
  </tr>
  <tr> 
    <th>Calculation Scenario</th>
    <th>Avg FC</th>
    <th>Max FC</th>
    <th>Min FC</th>
    <th>Avg Ret%</th>
  </tr>
"""
for r in forecasts:    
   email_body += generate_nested_html(r)
email_body += """
</table><p style="color: purple;">
 <strong> Please Note:</strong> Above stocks are filtered based on below criteria
  <ul>
 <li>Historical avg yearly returns > 12% </li> <li>CMP> Rs.50 </li> <li>Kurtosis lies between 2-4 </li>
 <li>Forecasted avg price based on both return and price difference forecast for next 6 month > 5% </li>
</ui></p>
<p><strong style="color:Tomato;"> Please See:</strong> Summary column tried to capture financial strength of the company with the help of Gemini AI. It may not give accurate picture.Please do the Fundamental analysis manually.</p>
</h2><br>
"""
email_body += f"""
      <p style="text-align:right;"><strong>Last updated on:</strong>{current_time_ist}</p>
      <div class="footer">
         <span class="page"></span></div>
    </div>
       </body>
       </html>"""

#generate_pdf(email_body,footer_html)
#pdfkit.from_string(email_body, 'Arima_forecast_summary.pdf')
output_pdf = "Arima_forecast_summary.pdf"
HTML(string=email_body).write_pdf(output_pdf)

def send_email(subject, html_content, receiver_emails, attachment_path=None):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = se
    sender_password = pwd

    # Create message container - the correct MIME type is multipart/alternative.
    msg = MIMEMultipart('alternative')
    msg['From'] = sender_email
    msg['To'] = ', '.join(receiver_emails)
    msg['Subject'] = subject

    # Attach HTML content
    msg.attach(MIMEText(html_content, 'html'))

    # Attach PDF file if path is provided
    if attachment_path:
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(attachment_path)}')
        msg.attach(part)

    # Send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
        print('Email sent successfully!')
    except Exception as e:
        print(f'Failed to send email. Error: {e}')

# Format email body as HTML table
e_body = """
<html>
<body>
  <h2>Hi All,<br>
  Stocks with their forecasted returns for the next 6 months </h2>
  </body></html>
"""
receiver_emails = re
#receiver_emails = ['sandeephs.rvim22@gmail.com']

# Path to the PDF file you want to attach
pdf_attachment_path ='Arima_forecast_summary.pdf'

# Send email with HTML content and PDF attachment
send_email(ms, e_body, receiver_emails, attachment_path=pdf_attachment_path)
