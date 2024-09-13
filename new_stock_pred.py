import pandas as pd
import yfinance as yf
from pmdarima.arima import auto_arima
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import kurtosis 
import optuna
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
from jinja2 import Environment, FileSystemLoader
import pdfkit
#from weasyprint import HTML, CSS
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.ERROR)

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
ms= 'Stock Forecast Results: '+current_time_ist
msp= ms+'.pdf'
start_date = five_years_ago.strftime('%Y-%m-%d')
end_date = current_date.strftime('%Y-%m-%d')

model = genai.GenerativeModel("models/gemini-1.0-pro")  

def calculate_pht(df,cmp):
  df = df.reset_index()
  df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)
  model = Prophet()
  model.fit(df)
  future = model.make_future_dataframe(periods=120, freq='B') 
  forecast = model.predict(future)
  minv = np.min(forecast[['yhat']].tail(120) ).round(2)
  maxv = np.max(forecast[['yhat']].tail(120) ).round(2)
  avgv = np.mean(forecast[['yhat']].tail(120) ).round(2)
  avgr = ((avgv-cmp)*100/cmp).round(2)
  return ["Prophet Model", minv,maxv,avgv,avgr]
    
def calculate_ema(values, window):
    return values.ewm(span=window, adjust=False).mean()
current_date = datetime.now().date()
five_years_ago = current_date - timedelta(days=5 * 365)
def calculate_stochastic_oscillator(df):
    n=14
    m=3
    df = df.copy()
    df['Lowest Low'] = df['Close'].rolling(window=n).min()
    df['Highest High'] = df['Close'].rolling(window=n).max()
    df['Stochastic_K'] = ((df['Close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low'])) * 100
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=m).mean()
    df.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)
    last_k = df['Stochastic_K'].iloc[-1]
    last_d = df['Stochastic_D'].iloc[-1]
    if last_k > last_d and 20 < last_k < 80:  
        return "Buy"
    else:
        return "Sell"

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

def testrn(cp,prd,log_returns):
   X_cp = cp.dropna().values.reshape(-1, 1)[:-1]  # Features (lagged log returns)
   y_cp = cp.dropna().values[1:]
   X_pd = prd.dropna().values.reshape(-1, 1)[:-1]  # Features (lagged log returns)
   y_pd = prd.dropna().values[1:]
   X_re = log_returns.dropna().values.reshape(-1, 1)[:-1]  # Features (lagged log returns)
   y_re = log_returns.dropna().values[1:]
   X_train_cp, X_test_cp, y_train_cp, y_test_cp = train_test_split(X_cp, y_cp, test_size=0.15, random_state=42)
   X_train_re, X_test_re, y_train_re, y_test_re = train_test_split(X_re, y_re, test_size=0.15, random_state=42)
   X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(X_pd, y_pd, test_size=0.15, random_state=42)
   return X_cp,y_cp,X_pd,y_pd,X_re,y_re,X_train_cp, X_test_cp, y_train_cp, y_test_cp,X_train_re, X_test_re, y_train_re, y_test_re,X_train_pd, X_test_pd, y_train_pd, y_test_pd
   

def objective(trial,X_cp,y_cp,X_pd,y_pd,X_re,y_re,X_train_cp, X_test_cp, y_train_cp, y_test_cp,X_train_re, X_test_re, y_train_re, y_test_re,X_train_pd, X_test_pd, y_train_pd, y_test_pd):
    mtt= ['RandomForest', 'GradientBoosting', 'XGB', 'NeuralNetwork']
    mt = trial.suggest_int('model_type',0,3)
    model_type = mtt[mt]
    mii= ['cp', 're', 'pd']
    mi= trial.suggest_int('model_inp',0, 2)
    model_inp = mii[mi]
    if model_type == 'RandomForest':
      n_estimators = trial.suggest_int('n_estimators', 100, 1000)
      max_depth = trial.suggest_int('max_depth', 5, 20)
      min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
      min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
      max_features = trial.suggest_float('max_features', 0.4, 1.0)
      model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
      )

    elif model_type == 'GradientBoosting':
      n_estimators = trial.suggest_int('n_estimators', 100, 1000)
      learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
      max_depth = trial.suggest_int('max_depth', 5, 20)
      min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
      min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
      max_features = trial.suggest_float('max_features', 0.5, 1.0)
      model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
     )

    elif model_type == 'XGB':
      n_estimators = trial.suggest_int('n_estimators', 100, 1000)
      learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
      max_depth = trial.suggest_int('max_depth', 5, 20)
      min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
      model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight
      )

    elif model_type == 'NeuralNetwork':
      hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (200,)])
      activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
      alpha = trial.suggest_float('alpha', 0.0001, 0.1)
      model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha
     )
    if model_inp == 'cp':
      model.fit(X_train_cp, y_train_cp)
      y_pred = model.predict(X_test_cp)
      mse = mean_squared_error(y_test_cp, y_pred)
      mape = mean_absolute_percentage_error(y_test_cp, y_pred)
    elif model_inp == 'pd':
      model.fit(X_train_pd, y_train_pd)
      y_pred = model.predict(X_test_cp)
      mse = mean_squared_error(y_test_pd, y_pred)
      mape = mean_absolute_percentage_error(y_test_pd, y_pred)
    elif model_inp == 're':
      model.fit(X_train_re, y_train_re)
      y_pred = model.predict(X_test_re)
      mse = mean_squared_error(y_test_re, y_pred)
      mape = mean_absolute_percentage_error(y_test_re, y_pred)

    return mse, mape


def ht(cp,prd,log_returns,cmp):
   for filename in os.listdir():
        if filename.endswith('_study.db'):
            os.remove(filename)
   study_name = 'a' + '_study'
   storage = 'sqlite:///' + 'a' + '_study.db'
   X_cp,y_cp,X_pd,y_pd,X_re,y_re,X_train_cp, X_test_cp, y_train_cp, y_test_cp,X_train_re, X_test_re, y_train_re, y_test_re,X_train_pd, X_test_pd, y_train_pd, y_test_pd = testrn(cp, prd,log_returns)
   study = optuna.create_study(directions=['minimize', 'minimize'], study_name=study_name, storage=storage, load_if_exists=True)
   study.optimize(lambda trial: objective(trial, X_cp,y_cp,X_pd,y_pd,X_re,y_re,X_train_cp, X_test_cp, y_train_cp, y_test_cp,X_train_re, X_test_re, y_train_re, y_test_re,X_train_pd, X_test_pd, y_train_pd, y_test_pd), n_trials=1500,n_jobs= -1) 
   best_trials = study.best_trials
   best_trial = best_trials[0]  
   print("Best hyperparameters:", best_trial.params)
   print("input type:",best_trial.params['model_inp'])
   if(best_trial.params['model_inp'] == 0):
      X = X_cp
      y = y_cp
      forecast_input = cp.dropna().values[-1].reshape(1, -1)  
      it = 'cp'
   elif(best_trial.params['model_inp'] == 1):
      X = X_pd
      y = y_pd
      forecast_input = prd.dropna().values[-1].reshape(1, -1)  
      it = 'pd'
   elif(best_trial.params['model_inp'] == 2):
      X = X_re
      y = y_re
      forecast_input = log_returns.dropna().values[-1].reshape(1, -1)  
      it = 're'
   params = best_trial.params.copy()
   del params['model_inp']
   if best_trial.params['model_type'] == 0:
     ru= "RandomForest model"
     del params['model_type']
     best_model = RandomForestRegressor(**params)
   elif best_trial.params['model_type'] == 1:
     ru=  "GradientBoosting model"
     del params['model_type']
     best_model = GradientBoostingRegressor(**params)
   elif best_trial.params['model_type'] == 2:
     ru= "XGB model"
     del params['model_type']
     best_model = xgb.XGBRegressor(**params)
   elif best_trial.params['model_type'] == 3:
     ru= "NeuralNetwork model"
     del params['model_type']
     best_model = MLPRegressor(**params)

   best_model.fit(X, y)
   yr = cmp
   forecast_returns = []
   for i in range(126):
     forecast_return = best_model.predict(forecast_input)
     ty= (forecast_return[0])
     forecast_input = np.array([[forecast_return[0]]])
     if it=='cp':
       forecast_returns.append(ty)
       yr = ty
     elif it=='pd':
       yr = ty+yr
       forecast_returns.append(yr)
     elif it=='re':
       yr = (1+ty)*yr
       forecast_returns.append(yr)
   mnf = (np.min(forecast_returns)).round(2)
   mxf = (np.max(forecast_returns)).round(2)
   avf = (np.mean(forecast_returns)).round(2)
   mnr = ((mnf-cmp)*100/cmp).round(2)
   mxr = ((mxf-cmp)*100/cmp).round(2)
   avr = ((avf-cmp)*100/cmp).round(2)
   return [ru, mnf,mxf, avf,avr]

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
        
       #pdfkit.from_string(html_content, 'Arima_forecast_summary.pdf', options=options)
       # print("PDF created successfully: Arima_forecast_summary.pdf")
    except Exception as e:
        print(f"Error creating PDF: {e}")
def fndmntl(ticker):
    stock = yf.Ticker(ticker)
    return stock   
    
def forecast_stock_returns(ticker_symbol):
    print(ticker_symbol)
    try:
      sck = yf.Ticker(ticker_symbol)
      stock_data = (sck.history(period='max')).dropna(inplace=True)
      if len(stock_data) > 1200:
        stock_data = stock_data.tail(1200)
      else:
        pass
      if len(stock_data)>500:
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
        if yearly_returns>9.9 and current_cmp>20 :
            z_scores = np.abs(stock_data['Returns'] - stock_data['Returns'].mean()) / stock_data['Returns'].std()
            stock_data = stock_data[(z_scores < 3)]  
            z_scores1 = np.abs(stock_data['Diff'] - stock_data['Diff'].mean()) / stock_data['Diff'].std()
            stock_data = stock_data[(z_scores1 < 3)]
            res_p01= calculate_pht(stock_data['Adj Close'],current_cmp)
            if res_p01[4]>5: 
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
              if res_p11[4]>3 and res_p21[4]>3:
                print("matching ") 
                res_p44= ht(stock_data['Adj Close'], stock_data['Diff'], stock_data['Returns'],current_cmp)
                if res_p44[4]> 5:  
                  a= fndmntl(ticker_symbol) 
                  query = "Read and summarize financial position/n"+ (((a.balance_sheet).iloc[:, :2]).dropna()).to_string() + "and "+(((a.financials).iloc[:, :2]).dropna()).to_string()+ "and "+(((a.financials).iloc[:, :2]).dropna()).to_string()
                  j=model.generate_content(query)
                  k= [ticker_symbol,v,[res_p01,res_p11,res_p12,res_p21,res_p22,res_p44],j.text,TI]
                  return k
                else:
                  return "NA"
              else:
                  return "NA"      
            else:
              return "NA"  
        else:
            return "NA"
      else:
           return "NA"
    except Exception as e:
         print(f"Error fetching data for {ticker_symbol}: {str(e)}")
         return "NA"


t =[]
forecasts = []
for ticker_symbol in ticker_symbols:
    result = forecast_stock_returns(ticker_symbol)
    if result != "NA":
      forecasts.append(result)

def generate_nested_html(single_row): 
    nested_html = "<tr>"
    nested_html += f"<td>{single_row[0]}</td>"
    
    nested_html += "<td>"
    nm =["Yearly Ret%","CMP","52 Wk High","52 Wk Low","RSI"]  
    nested_html += "<table border='1'>\n"
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
<h1 style="text-align:center;color: #440000;"> <u> Weekly Forecast Summary NSE stocks using Prophet, ARIMA and NeuralNetwork/XGB/RandomForest/GradientBoosting </u></h1>
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
 <li>Historical avg yearly returns > 12% </li> <li>CMP> Rs.20 </li> 
 <li> Stocks with good fundamentals data </li>
 <li>Forecasted avg price based on both return and price difference forecast for next 6 month > 3% </li><li> Average returns as per prophet model for next 6 month > 5%</li>
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
pdfkit.from_string(email_body, msp)
#output_pdf = "Arima_forecast_summary.pdf"
#HTML(string=email_body).write_pdf(output_pdf)

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

# Path to the PDF file you want to attach
pdf_attachment_path = msp

# Send email with HTML content and PDF attachment
send_email(ms, e_body, receiver_emails, attachment_path=pdf_attachment_path)
