import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from datetime import datetime, timedelta

current_time_ist = (datetime.now() + timedelta(hours=5, minutes=30, seconds=0)).strftime("%Y-%m-%d %H:%M:%S") 
current_date = datetime.now().date()
five_years_ago = current_date - timedelta(days=5 * 365)

start_date = five_years_ago.strftime('%Y-%m-%d')
end_date = current_date.strftime('%Y-%m-%d')

data = (yf.download('TCS.NS', start=start_date, end=end_date)['Close']).dropna()
# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Define Bayesian Optimization
pbounds = {
    'lstm_units': (50, 200),
    'gru_units': (20, 200),
    'dropout_rate': (0.1, 0.5),
    'batch_size': (32, 128),
    'optimizer_idx': (0, 2),
    'window_size': (120, 150)
}

def optimize_model(lstm_units, gru_units, dropout_rate, batch_size, optimizer_idx, window_size):
    X, y = [], []
    for i in range(len(scaled_data) - int(window_size)):
        X.append(scaled_data[i:i + int(window_size)])
        y.append(scaled_data[i + int(window_size)])
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Bidirectional(LSTM(int(lstm_units), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(GRU(int(gru_units), return_sequences=True)))
    model.add(Dense(1))
    model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(optimizer_idx)], loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, epochs=50, batch_size=int(batch_size), validation_split=0.2, callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    return {
        'loss': -mae,
        'mse': mse,
        'mape': mape,
        'r2': r2
    }

optimizer = BayesianOptimization(
    f=optimize_model,
    pbounds=pbounds,
    random_state=0,
    verbose=2,
    multi_objective=True,
    objectives=[
        {'name': 'mae', 'type': 'minimize', 'goal': 0.0},
        {'name': 'mse', 'type': 'minimize', 'goal': 0.0},
        {'name': 'mape', 'type': 'minimize', 'goal': 0.0},
        {'name': 'r2', 'type': 'maximize', 'goal': 1.0}
    ]
)

# Run the optimization with a timeout
start_time = time.time()
timeout = 30 * 60  # 1 hour
while True:
    try:
        optimizer.maximize(init_points=5, n_iter=10)
        break
    except Exception as e:
        print(f"Error: {e}")
        if time.time() - start_time > timeout:
            print("Timeout reached. Stopping optimization.")
            break

# Get the best parameters
best_params = optimizer.max

# Train model with best parameters
X, y = [], []
for i in range(len(scaled_data) - int(best_params['params']['window_size'])):
    X.append(scaled_data[i:i + int(best_params['params']['window_size'])])
    y.append(scaled_data[i + int(best_params['params']['window_size'])])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Bidirectional(LSTM(int(best_params['params']['lstm_units']), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(best_params['params']['dropout_rate']))
model.add(Bidirectional(GRU(int(best_params['params']['gru_units']), return_sequences=True)))
model.add(Dense(1))
model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(best_params['params']['optimizer_idx'])], loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=int(best_params['params']['batch_size']), validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Forecast
forecasted = []
current_data = scaled_data[-int(best_params['params']['window_size']):]
for _ in range(126):
    prediction = model.predict(current_data.reshape(1, int(best_params['params']['window_size']), 1))[0, 0]
    forecasted.append(prediction)
    current_data = np.append(current_data[1:], prediction)

# Scale back the forecasted prices
forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))
print('forecasted_prices:', forecasted_prices)

# Plot forecasted prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data.values, label='Historical Data')
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=126)
plt.plot(forecast_dates, forecasted_prices, label='Forecasted Data', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TCS.NS Stock Price Forecast')
plt.legend()
plt.savefig('plot.png')
