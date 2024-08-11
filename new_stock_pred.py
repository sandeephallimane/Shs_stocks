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
import sqlite3
from weasyprint import HTML, CSS
import os
import google.generativeai as genai

ta=0
def tab():
  conn = sqlite3.connect('sandeephallimane.db')
  cursor = conn.cursor()
  cursor.execute('''CREATE TABLE IF NOT EXISTS stock_analysis (
    ticker TEXT PRIMARY KEY,
    yearly_returns REAL NOT NULL,
    current_cmp REAL NOT NULL,
    stock_data_52_high REAL NOT NULL,
    stock_data_52_low REAL NOT NULL,
    rsi REAL NOT NULL,
    Return_Non_Seasonal_max REAL NOT NULL,
    Return_Non_Seasonal_min REAL NOT NULL,
    Return_Non_Seasonal_avg REAL NOT NULL,
    Return_Non_Seasonal_ret REAL NOT NULL,
    Return_Seasonal_max REAL NOT NULL,
    Return_Seasonal_min REAL NOT NULL,
    Return_Seasonal_avg REAL NOT NULL,
    Return_Seasonal_ret REAL NOT NULL,
    Price_Diff_Non_Seasonal_max REAL NOT NULL,
    Price_Diff_Non_Seasonal_min REAL NOT NULL,
    Price_Diff_Non_Seasonal_avg REAL NOT NULL,
    Price_Diff_Non_Seasonal_ret REAL NOT NULL,
    Price_Diff_Seasonal_max REAL NOT NULL,
    Price_Diff_Seasonal_min REAL NOT NULL,
    Price_Diff_Seasonal_avg REAL NOT NULL,
    Price_Diff_Seasonal_ret REAL NOT NULL,
    LSTM_max REAL NOT NULL,
    LSTM_min REAL NOT NULL,
    LSTM_avg REAL NOT NULL,
    LSTM_ret REAL NOT NULL,
    summary TEXT NOT NULL,
    rsi_ind TEXT NOT NULL,
    macd TEXT NOT NULL,
    ema TEXT NOT NULL,
    stochastic TEXT NOT NULL,
    status TEXT NOT NULL
)''')
  conn.commit()
  conn.close()

ticker_symbols=(os.getenv('TS')).split(',')
print(ticker_symbols)
ak= os.getenv('AK')
if ak is None :
    raise ValueError("API Key not found in environment variables")

genai.configure(api_key=ak)

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
      stock_data = yf.download(ticker_symbol, period='5y').dropna()
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
        if yearly_returns>12 and current_cmp>20:
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
            if res_p11[4]>5 and res_p21[4]>5:
               k= [ticker_symbol,v[0],v[1],v[2],v[3],v[4],res_p11[1],res_p11[2],res_p11[3],res_p11[4],res_p12[1],res_p12[2],res_p12[3],res_p12[4],res_p21[1],res_p21[2],res_p21[3],res_p21[4],res_p22[1],res_p22[2],res_p22[3],res_p22[4],0,0,0,0,"NA",TI[0],TI[1],TI[2],TI[3],"N"]
               return k
            else:
               return "NA"
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
        a= fndmntl(result[0])
        query = "Read and summarize financial position/n"+ (((a.balance_sheet).iloc[:, :2]).dropna()).to_string() + "and "+(((a.financials).iloc[:, :2]).dropna()).to_string()+ "and "+(((a.financials).iloc[:, :2]).dropna()).to_string()
        model = genai.GenerativeModel("models/gemini-1.0-pro")  
        j=model.generate_content(query)
        result[26] = j.text
        if(ta=0):
            tab()
            ta = ta+1
        conn = sqlite3.connect('sandeephallimene.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO stock_analysis (
    ticker,
    yearly_returns,
    current_cmp,
    stock_data_52_high,
    stock_data_52_low,
    rsi,
    Return_Non_Seasonal_max,
    Return_Non_Seasonal_min,
    Return_Non_Seasonal_avg,
    Return_Non_Seasonal_ret,
    Return_Seasonal_max,
    Return_Seasonal_min,
    Return_Seasonal_avg,
    Return_Seasonal_ret,
    Price_Diff_Non_Seasonal_max,
    Price_Diff_Non_Seasonal_min,
    Price_Diff_Non_Seasonal_avg,
    Price_Diff_Non_Seasonal_ret,
    Price_Diff_Seasonal_max,
    Price_Diff_Seasonal_min,
    Price_Diff_Seasonal_avg,
    Price_Diff_Seasonal_ret,
    LSTM_max,
    LSTM_min,
    LSTM_avg,
    LSTM_ret,
    summary,
    rsi_ind,
    macd,
    ema,
    stochastic,
    status
) VALUES (?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', result)
        result= []
        conn.commit()
        conn.close()
