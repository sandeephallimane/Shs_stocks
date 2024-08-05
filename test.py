import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt

# Load data
current_time_ist = (pd.Timestamp.now() + pd.Timedelta(hours=5, minutes=30, seconds=0)).strftime("%Y-%m-%d %H:%M:%S") 
current_date = pd.Timestamp.now().date()
five_years_ago = current_date - pd.Timedelta(days=5 * 365)

start_date = five_years_ago.strftime('%Y-%m-%d')
end_date = current_date.strftime('%Y-%m-%d')

data = (yf.download('TCS.NS', start=start_date, end=end_date)['Close']).dropna()

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

def optimize_model(trial):
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    gru_units = trial.suggest_int('gru_units', 20, 200)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    optimizer_idx = trial.suggest_int('optimizer_idx', 0, 2)
    window_size = trial.suggest_int('window_size', 120, 150)

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

    return mae, mse, mape, r2

study = optuna.create_study(directions=['minimize', 'minimize', 'minimize', 'maximize'])
study.optimize(optimize_model, n_trials=20)

best_trial = study.best_trial
print('Best parameters:', best_trial.params)
print('Best objective values:', best_trial.values)

# Train model with best parameters
X, y = [], []
for i in range(len(scaled_data) - int(best_trial.params['window_size'])):
    X.append(scaled_data[i:i + int(best_trial.params['window_size'])])
    y.append(scaled_data[i + int(best_trial.params['window_size'])])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Bidirectional(LSTM(int(best_trial.params['lstm_units']), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(best_trial.params['dropout_rate']))
model.add(Bidirectional(GRU(int(best_trial.params['gru_units']), return_sequences=True)))
model.add(Dense(1))
model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(best_trial.params['optimizer_idx'])], loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=int(best_trial.params['batch_size']), validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Forecast
forecasted = []
current_data = scaled_data[-int(best_trial.params['window_size']):]
for _ in range(126):
    prediction = model.predict(current_data.reshape(1, int(best_trial.params['window_size']), 1))[0, 0]
    forecasted.append(prediction)
    current_data = np.append(current_data[1:], prediction)

# Scale back the forecasted prices
forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))
print('Forecasted prices:', forecasted_prices)

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
