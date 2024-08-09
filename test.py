import optuna
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

class OptunaStudy:
    def __init__(self, study_name, storage):
        self.study = optuna.create_study(study_name=study_name, storage=storage)

    def optimize(self, objective_func, n_trials):
        self.study.optimize(objective_func, n_trials=n_trials)

    def get_best_trial(self):
        return self.study.best_trial

    def get_best_params(self):
        return self.study.best_params

def create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size):
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))
    model.add(Bidirectional(LSTM(int(lstm_units), return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(GRU(int(gru_units), return_sequences=True)))
    model.add(Dense(1))
    optimizers = ['adam', 'rmsprop', 'sgd']
    model.compile(optimizer=optimizers[int(optimizer_idx)], loss='mean_squared_error')
    return model

def optimize_model(trial, scaled_data):
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    gru_units = trial.suggest_int('gru_units', 20, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
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
    history = model.fit(X_train, y_train, epochs=20, batch_size=int(batch_size), validation_split=0.2, callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_test)[:, -1, :]
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if y_test.all() != 0 else 100
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    ic = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
    aic = len(y_test) * np.log(mse) + 2 * model.count_params()
    msle = np.mean((np.log(y_test + 1) - np.log(y_pred + 1)) ** 2)
    mad = np.mean(np.abs(y_test - y_pred))
    mfe = np.mean((y_test - y_pred) / y_test) * 100 if y_test.all() != 0 else 100
    return mae, mse, rmse, mape, r2, ic, aic, msle, mad, mfe

def new_lstm(ti, scaled_data, scaler, lst):
    for filename in os.listdir():
       if filename.endswith('_study.db'):
         os.remove(filename)
    study_name = ti + '_study'
    storage = 'sqlite:///' + study_name + '.db'
    study = OptunaStudy(study_name, storage)
    study.optimize(lambda trial: optimize_model(trial, scaled_data), n_trials=5)

    best_trial = study.get_best_trial()
    best_params = study.get_best_params()

    best_model = create_model(**best_params)
    window_size = int(best_params['window_size'])
    bts = int(best_params['batch_size'])

    scaled_data = scaled_data[~np.isnan(scaled_data).any(axis=1)] 
    scaled_data = scaled_data[scaled_data != None]  
    scaled_data = scaled_data.reshape(len(scaled_data), 1, 1)
    num_windows = len(scaled_data) // window_size
    scaled_data = scaled_data[-num_windows * window_size:]
    scaled_data = scaled_data.reshape(-1, window_size, 1)
    
    # Check for None values
    if np.any(scaled_data == None):
        raise ValueError("None values found in scaled_data")
    
    best_model.fit(scaled_data, epochs=100, batch_size=bts, verbose=0)

    last_date = lst
    forecast_dates = pd.date_range(start=last_date, periods=126, freq='D')
    forecasted_prices = []
    current_data = scaled_data[-int(best_params['window_size']):]
    for date in forecast_dates:
        prediction = best_model.predict(current_data.reshape(1, int(best_params['window_size']), 1))[:, -1, :]
        forecasted_prices.append(prediction[0, 0])
        current_data = np.append(current_data[1:], prediction[0, 0])
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))
    return forecasted_prices

tickers = ['TCS.NS', 'INFY.NS']

for t in tickers:
    scaler = MinMaxScaler()
    scaled_data, lst = stk_dt(t, scaler)
    f = new_lstm(t, scaled_data, scaler, lst)
    print("Stock name:", t)
    print("Forecasted prices:", f)
