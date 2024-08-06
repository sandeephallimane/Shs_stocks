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
import os

def delete_all_studies():
    db_file = 'optuna.db'  # or your custom database file
    if os.path.exists(db_file):
        os.remove(db_file)
            

tickers = ['TCS.NS','INFY.NS']

def new_lstm(ti):
  delete_all_studies()
  script_name= ti
  study_name = script_name + '_study'
  storage = 'sqlite:///' + script_name + '_study.db'
  def optimize_model(trial):
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    gru_units = trial.suggest_int('gru_units', 20, 200)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    optimizer_idx = trial.suggest_int('optimizer_idx', 0, 2)
    window_size = trial.suggest_int('window_size', 100, 200)

    X, y = [], []
    for i in range(len(scaled_data) - int(window_size)):
        X.append(scaled_data[i:i + int(window_size)])
        y.append(scaled_data[i + int(window_size)])
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, epochs=50, batch_size=int(batch_size), validation_split=0.2, callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_test)[:, -1, :]
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if y_test.all() != 0 else 100
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    return mae, mse, rmse, mape, r2
    
  def create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size):
      model = Sequential()
      model.add(Bidirectional(LSTM(int(lstm_units), return_sequences=True, input_shape=(window_size, 1))))
      model.add(Dropout(dropout_rate))
      model.add(Bidirectional(GRU(int(gru_units), return_sequences=True)))
      model.add(Dense(1))
      model.compile(optimizer=['adam', 'rmsprop', 'sgd'][int(optimizer_idx)], loss='mean_squared_error')
      return model

  data = yf.download(script_name, period='5y')['Close'].dropna()
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
  n_jobs = -1
  study = optuna.create_study(directions=['minimize', 'minimize', 'minimize', 'minimize', 'maximize'],
                            study_name=study_name,
                            storage=storage,
                            load_if_exists=True)
  study.optimize(optimize_model, n_trials=50, n_jobs=n_jobs)
  best_trials = study.best_trials
  best_trial = best_trials[0]
  best_model = create_model(**best_trial.params)
  X, y = [], []
  for i in range(len(scaled_data) - int(best_trial.params['window_size'])):
     X.append(scaled_data[i:i + int(best_trial.params['window_size'])])
     y.append(scaled_data[i + int(best_trial.params['window_size'])])
  X, y = np.array(X), np.array(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  best_model.fit(X_train, y_train, epochs=50, batch_size=int(best_trial.params['batch_size']), validation_split=0.2, verbose=0)
  forecasted = []
  current_data = scaled_data[-int(best_trial.params['window_size']):]
  for _ in range(126):
     prediction = best_model.predict(current_data.reshape(1, int(best_trial.params['window_size']), 1))[:, -1, :]
     forecasted.append(prediction[0, 0])
     current_data = np.append(current_data[1:], prediction[0, 0])
  forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))
  return forecasted_prices

for t in tickers:
    f=new_lstm(t)
    print("Stock name:",t)
    print("forecasted_prices:",f)
