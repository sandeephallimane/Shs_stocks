import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, GRU
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError,Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError,MeanAbsolutePercentageError , MeanSquaredLogarithmicError
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import optuna
import sqlite3
from tensorflow.keras import regularizers
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from keras.regularizers import l1
from keras.layers import BatchNormalization
from keras.activations import relu, leaky_relu, swish
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import TimeSeriesSplit
import requests
import ast

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

url = "https://raw.githubusercontent.com/sandeephallimane/Shs_stocks/main/data.txt"
response = requests.get(url)
if response.status_code == 200:
    data = response.text
    lines = data.splitlines()
    a = len(lines)
else:
    print("Failed to retrieve file")


def download_file(url, filename):
    try:
        response = requests.get(url)
        
        # If the file exists, save it to the local directory
        if response.status_code == 200:
            with open(filename, 'w') as file:
                file.write(response.text)
            data = response.text
            lines = data.splitlines()
            b = len(lines)
            print(f"File downloaded successfully: {filename}")
            return b
        else:
            # If the file does not exist, create an empty file
            with open(filename, 'w') as file:
                pass
            print(f"File not found. Created empty file: {filename}")
            b= 0
            return b
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

url2 = "https://raw.githubusercontent.com/sandeephallimane/Shs_stocks/main/upddata.txt"
filename2 = "upddata.txt"


early_stopping = EarlyStopping(monitor='loss', patience=5 )

def select_loss_function(scaled_data):
    mean = np.mean(scaled_data)
    std = np.std(scaled_data)
    range = np.ptp(scaled_data)
    skewness = stats.skew(scaled_data)
    kurtosis = stats.kurtosis(scaled_data)
    mean_squared_error = MeanSquaredError()
    mean_absolute_error = MeanAbsoluteError()
    huber_loss = Huber()
    mean_squared_logarithmic_error = MeanSquaredLogarithmicError()
    
    if mean < 0.1 and std < 0.1:   
        return 'mean_squared_error'
    elif mean < 1 and std < 1:  
        return 'mean_absolute_error'
    elif range > 1 and skewness > 1:  
        return 'huber_loss'
    elif kurtosis > 1:  
        return 'mean_squared_logarithmic_error'
    else:  
        return 'mean_absolute_error' 
        
def stk_dt(tk,scaler):
   data = yf.download(tk, period='5y')['Close'].dropna()
   last_date = pd.to_datetime(data.index[-1].to_pydatetime().date())
   scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
   cmp = data.iloc[-1].round(2)
   return scaled_data,last_date,cmp

def create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size, activation, loss_function):
    model = Sequential()
    model.add(Bidirectional(LSTM(int(lstm_units), return_sequences=True, input_shape=(window_size, 1), activation=activation)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(GRU(int(gru_units), return_sequences=True, activation=activation)))
    model.add(Dense(1, kernel_regularizer=l1(0.01)))
    model.compile(
        optimizer=['adamw', 'nadam', 'adam'][int(optimizer_idx)],
        loss=loss_function,
        metrics=[
            'mean_absolute_error', 
            'mean_squared_error', 
            'mean_absolute_percentage_error', 
            'mean_squared_logarithmic_error'        ])
    return model

def optimize_model(trial, scaled_data, lf):
    lstm_units = trial.suggest_int('lstm_units', 100, 200)
    gru_units = trial.suggest_int('gru_units', 100, 150)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 32, 64)
    optimizer_idx = trial.suggest_int('optimizer_idx', 0, 2)
    window_size = trial.suggest_int('window_size', 100, 150)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'swish'])
    activation = {'relu': relu, 'leaky_relu': leaky_relu, 'swish': swish}[activation]

    X, y = [], []
    for i in range(len(scaled_data) - int(window_size)):
        X.append(scaled_data[i:i + int(window_size)])
        y.append(scaled_data[i + int(window_size)])
    X, y = np.array(X), np.array(y)
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    model = create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size, activation, lf)
    history = model.fit(X_train, y_train, epochs=20, batch_size=int(batch_size), validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    mae = history.history['val_mean_absolute_error'][-1]
    mse = history.history['val_mean_squared_error'][-1]
    mape = history.history['val_mean_absolute_percentage_error'][-1]
    rmse =  np.sqrt(mse)
    msle = history.history['val_mean_squared_logarithmic_error'][-1]
    return mae, mse, rmse, msle, mape

def new_lstm(ti, scaled_data, scaler, lst, cmp):
    script_name= ti
    study_name = script_name + '_study'
    storage = 'sqlite:///' + script_name + '_study.db'

    lf= select_loss_function(scaled_data)
    sampler = TPESampler()   # or RandomSampler(),GridSampler() 
    study = optuna.create_study(directions=['minimize', 'minimize','minimize','minimize', 'minimize'], study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler)
    study.optimize(lambda trial: optimize_model(trial, scaled_data, lf), n_trials=30, n_jobs=8)
    best_trials = study.best_trials
    best_trial = best_trials[0]  
    best_model = create_model(**best_trial.params, loss_function=lf)
    print("best_model.summary:",best_model.summary()) 
    X, y = [], []
    for i in range(len(scaled_data) - int(best_trial.params['window_size'])):
      X.append(scaled_data[i:i + int(best_trial.params['window_size'])])
      y.append(scaled_data[i + int(best_trial.params['window_size'])])
    X, y = np.array(X), np.array(y)
    best_model.fit(X, y, epochs=100, batch_size=int(best_trial.params['batch_size']), callbacks=[early_stopping], verbose=0)
    best_model.summary()
    last_date = lst
    forecast_dates = pd.date_range(start=last_date, periods=126, freq='D')
    forecasted_prices = []
    window_size = int(best_trial.params['window_size'])
    current_data = scaled_data[-window_size:]
    for date in forecast_dates:
      current_data_reshaped = current_data.reshape(1, window_size, 1)
      prediction = best_model.predict(current_data_reshaped)
      forecasted_price = prediction[0, -1, 0]  # Check model output shape
      forecasted_prices.append(forecasted_price)
      current_data = np.append(current_data[1:], forecasted_price)
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))
    min_p = np.min(forecasted_prices).round(2)
    max_p = np.max(forecasted_prices).round(2)
    avg_p = np.mean(forecasted_prices).round(2)
    ret_p = ((avg_p-cmp)*100/cmp).round(2)
    return min_p,max_p,avg_p,ret_p


b= download_file(url2, filename2)
if a>b:
  t = ast.literal_eval(lines[b])
  print("Stock name:", t[0])
  scaler = MinMaxScaler()
  scaled_data, lst, cmp = stk_dt(t[0], scaler)
  t[22], t[23], t[24], t[25] = new_lstm(t[0], scaled_data, scaler, lst, cmp)
  t[31] = 'Y'
  print("Forecasted prices:", t[22], t[23], t[24], t[25])
  with open('upddata.txt', 'a') as f:
    f.write(str(t) + '\n')
print(a)
print(b)
os.environ['a'] = str(a)
os.environ['b'] = str(b)

