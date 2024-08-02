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

# Load data
data = (yf.download('TCS.NS', start='2019-01-01', end='2024-08-01')['Close']).dropna()
# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
X, y = [], []
for i in range(len(scaled_data) - 100):
    X.append(scaled_data[i:i + 100])
    y.append(scaled_data[i + 100])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(100, return_sequences=True)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Define Bayesian Optimization
pbounds = {
    'lstm_units': (50, 200),
    'gru_units': (20, 200),
    'dropout_rate': (0.1, 0.5),
    'batch_size': (32, 128),
    'optimizer_idx': (0, 2)
}

def optimize_model(lstm_units, gru_units, dropout_rate, batch_size, optimizer_idx):
    model.layers[0].units = int(lstm_units)
    model.layers[2].units = int(gru_units)
    model.layers[1].rate = dropout_rate
    model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(optimizer_idx)], loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, epochs=50, batch_size=int(batch_size), validation_split=0.2, callbacks=[early_stopping], verbose=0)
    score = model.evaluate(X_train, y_train, verbose=0)
    return -score

optimizer = BayesianOptimization(
    f=optimize_model,
    pbounds=pbounds,
    random_state=0,
    verbose=2
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
model.layers[0].units = int(best_params['params']['lstm_units'])
model.layers[2].units = int(best_params['params']['gru_units'])
model.layers[1].rate = best_params['params']['dropout_rate']
model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(best_params['params']['optimizer_idx'])], loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=int(best_params['params']['batch_size']), validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Forecast
forecasted = []
current_data = scaled_data[-100:]
for _ in range(126):
    prediction = model.predict(current_data.reshape(1, 100, 1))[0, 0]
    forecasted.append(prediction)
    current_data = np.append(current_data[1:], prediction)

# Scale back the forecasted prices
forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))

# Plot forecasted prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data.values, label='Historical Data')
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=126)
plt.plot(forecast_dates, forecasted_prices, label='Forecasted Data', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TCS.NS Stock Price Forecast')
