import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import requests
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
stock_list=(os.getenv('TS')).split(',')
ak= os.getenv('AK')
se=os.getenv('SE')
ma = os.getenv('MGA') 
md = os.getenv('MGD') 
GAS_URL = os.getenv('GAS')
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

MODELS = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, objective='reg:squarederror', random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
}

def get_stock_data(ticker, period="3y"):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty or len(df) < 250:
        raise ValueError(f"Not enough data for {ticker}")
    
    df["Return"] = df["Close"].pct_change()
    df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["Future_Price"] = df["Close"].shift(-120).rolling(120).mean()
    df.dropna(inplace=True)
    return df
    
def safe_process(ticker):
    try:
        return process_stock(ticker)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None
        
def train_best_model(df):
    features = ["RSI", "MACD", "Signal"]
    target = "Future_Price"
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        next_pred = model.predict([X_scaled[-1]])[0]
        results[name] = {"model": model, "mse": mse, "future_price_pred": next_pred}

    best_model_name = min(results, key=lambda k: results[k]["mse"])
    return best_model_name, results[best_model_name]["future_price_pred"]

def process_stock(ticker):
    try:
        df = get_stock_data(ticker)
        model_name, future_price_pred = train_best_model(df)
        current_price = df["Close"].iloc[-1]
        expected_return = (future_price_pred - current_price) / current_price
        return {
            "Ticker": ticker,
            "Model": model_name,
            "Expected Return": expected_return
        }
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None


def forecast_top_stocks(stock_list, top_n=20, n_jobs=4):
    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        future_to_ticker = {executor.submit(safe_process, ticker): ticker for ticker in stock_list}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res:
                results.append(res)

    sorted_results = sorted(results, key=lambda x: x["Expected Return"], reverse=True)
    return sorted_results[:top_n]


def build_portfolio(top_stocks):
    total_weight = sum(abs(stock["Expected Return"]) for stock in top_stocks)
    portfolio = []
    for stock in top_stocks:
        portfolio.append({
            "Ticker": stock["Ticker"],
            "Model": stock["Model"],
            "Expected Return (%)": round(stock["Expected Return"] * 100, 2),
            "Weight (%)": round((abs(stock["Expected Return"]) / total_weight) * 100, 2)
        })
    return pd.DataFrame(portfolio)

def generate_chart(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["Ticker"], df["Weight (%)"], color="#4a90e2")
    ax.set_title("Portfolio Allocation by Weight (%)")
    ax.set_ylabel("Weight (%)")
    ax.set_xlabel("Ticker")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" alt="Portfolio Chart" style="width:100%;max-width:800px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);"/>'

def build_html(df, chart_html):
    table_html = df.to_html(index=False, justify='center', border=0, classes='styled-table', escape=False)

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #f9f9f9;
                padding: 20px;
                color: #333;
            }}
            h2 {{
                color: #2c3e50;
            }}
            .styled-table {{
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 16px;
                width: 100%;
                background-color: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .styled-table th, .styled-table td {{
                padding: 12px 15px;
                text-align: center;
            }}
            .styled-table thead {{
                background-color: #4a90e2;
                color: #fff;
                text-transform: uppercase;
            }}
            .styled-table tbody tr:nth-child(even) {{
                background-color: #f3f3f3;
            }}
        </style>
    </head>
    <body>
        <h2>📊 6-Month Stock Forecast Portfolio</h2>
        {chart_html}
        <br/>
        {table_html}
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    top_n = 25
    top_stocks = forecast_top_stocks(stock_list, top_n=top_n, n_jobs=4)
    portfolio_df = build_portfolio(top_stocks)
    chart = generate_chart(portfolio_df)
    html = build_html(portfolio_df, chart)
    response = requests.post(GAS_URL, data={"html": html, "ty": "SF"})
    print(response.text)
  
