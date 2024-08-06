Here's the updated code with vectorization, improved hyperparameter tuning, and additional model evaluation metrics:
Python
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

data = yf.download('TCS.NS', period='5y')['Close'].dropna()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

train_size = int(0.8 * len(scaled_data))

def create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(int(lstm_units), return_sequences=True, input_shape=(1, 1))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(GRU(int(gru_units), return_sequences=True)))
    model.add(Dense(1))
    model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(optimizer_idx)], loss='mean_squared_error')
    return model

@tf.function(reduce_retracing=True)
def optimize_model(trial):
    window_size = trial.suggest_int('window_size', 120, 150)
    X_train, X_val = scaled_data[:train_size], scaled_data[train_size:]
    y_train, y_val = scaled_data[window_size:train_size+window_size], scaled_data[train_size+window_size:]

    X, y = [], []
    for i in range(len(X_train) - window_size):
        X.append(X_train[i:i + window_size])
        y.append(y_train[i + window_size])
    X, y = np.array(X), np.array(y)

    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    gru_units = trial.suggest_int('gru_units', 20, 200)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    optimizer_idx = trial.suggest_int('optimizer_idx', 0, 2)

    model = create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X, y, epochs=50, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_val)[:, -1, :]
    mae = np.mean(np.abs(y_val - y_pred))
    mse = np.mean((y_val - y_pred) ** 2)
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
    r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
    rmse = np.sqrt(mse)
    return mae, mse, mape, r2, rmse

study = optuna.create_study(directions=['minimize', 'minimize', 'minimize', 'maximize', 'minimize'])
study.optimize(optimize_model, n_trials=2)

best_trials = study.best_trials
best_trial = best_trials[0]
best_model = create_model(**best_trial.params)

# Train the best model on the entire dataset
X, y = [], []
for i in range(len(scaled_data) - int(best_trial.params['window_size'])):
    X.append(scaled_data[i:i + int(best_trial.params['window_size'])])
    y.append(scaled_data[i + int(best_trial.params['window_size'])])
X, y = np.array(X), np.array(y)
best_model.fit(X, y, epochs=50, batch_size=int(best_trial.params['batch_size']), validation_split=0.2, verbose=0)

forecast_dataset = tf.data.Dataset.from_tensor_slices(scaled_data[-int(best_trial.params['window_size']):])
forecast_dataset = forecast_dataset.batch(32)  

forecasts = best_model.predict(forecast_dataset)

forecasted = []
current_data = scaled_data[-int(best_trial.params['window_size']):]
for i in range(126):
    forecast = forecasts[i]
    forecasted.append(forecast[0, 0])
    current_data = np.append(current_data[1:], forecast[0, 0])

forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))
plt.figure(figsize=(14, 7))
plt.plot(data.index, data.values, label='Historical Data')
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=126)
plt.plot(forecast_dates, forecasted_prices, label='Forecasted Data', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TCS.NS Stock Price Forecast')
plt.legend()
plt.savefig('plot.png')

# Evaluate the model using cross-validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
mae_cv = []
mse_cv = []
mape_cv = []
r2_cv = []
rmse_cv = []
for train_index, val_index in tscv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    best_model.fit(X_train, y_train, epochs=50, batch_size=int(best_trial.params['batch_size']), verbose=0)
    y_pred = best_model.predict(X_val)[:, -1, :]
    mae_cv.append(np.mean(np.abs(y_val - y_pred)))
    mse_cv.append(np.mean((y_val - y_pred) ** 2))
    mape_cv.append(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100)
    r2_cv.append(1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)))
    rmse_cv.append(np.sqrt(np.mean((y_val - y_pred) ** 2)))

print('Cross-validation metrics:')
print('MAE:', np.mean(mae_cv))
print('MSE:', np.mean(mse_cv))
print('MAPE:', np.mean(mape_cv))
print('R2:', np.mean(r2_cv))
print('RMSE:', np.mean(rmse_cv))
