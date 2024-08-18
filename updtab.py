import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError,Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError,MeanAbsolutePercentageError , MeanSquaredLogarithmicError
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, AdamW, Nadam
import optuna
import sqlite3
from optuna.trial import Trial
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import regularizers
from optuna import create_study
from optuna.samplers import TPESampler
from keras.regularizers import l1,l2
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

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

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
        
def stk_dt(tk):
   data = yf.download(tk, period='5y')['Close'].dropna()
   return data

mean_squared_error = MeanSquaredError()
mean_absolute_error = MeanAbsoluteError()
huber_loss = Huber()
loss_functions = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.Huber()]
loss_function_categories = list(range(len(loss_functions)))
loss_categories = ['mse', 'mae', 'huber']
loss_functions_dict = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError(),
        'huber': tf.keras.losses.Huber()
    }
def create_model(trial, window_size, loss_functions):
    loss_name = trial.suggest_categorical('loss_function', loss_categories)
    loss = loss_functions_dict[loss_name]
    print(loss)
    model = Sequential()
    model.add(Bidirectional(LSTM(int(trial.suggest_int('lstm_units', 50, 200)), 
                                  return_sequences=True, 
                                  input_shape=(window_size, 1), 
                                  activation=trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'swish', 'tanh']), 
                                  dropout=trial.suggest_float('dropout_rate', 0.1, 0.5), 
                                  recurrent_dropout=trial.suggest_float('recurrent_dropout',0.1, 0.2))))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout_rate', 0.1, 0.5)))
    model.add(Bidirectional(GRU(int(trial.suggest_int('gru_units', 50, 200)), 
                                  return_sequences=True, 
                                  activation=trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'swish', 'tanh']), 
                                  dropout=trial.suggest_float('dropout_rate', 0.1, 0.5), 
                                  recurrent_dropout=trial.suggest_float('recurrent_dropout', 0.1, 0.2))))
    model.add(Dropout(trial.suggest_float('dropout_rate', 0.1, 0.5)))
    model.add(Dense(1, kernel_regularizer=l2(trial.suggest_float('l2', 0.01, 0.1))))
    optimizers = [Adam(), RMSprop(), SGD(), AdamW(), Nadam()]
    model.compile(
        optimizer=optimizers[trial.suggest_int('optimizer_idx', 0, 4)],
        loss=loss,
        metrics=['mean_squared_error', 'mean_absolute_error']
    )
    
    return model

def optimize_model(trial, scaled_data):
    window_size = trial.suggest_int('window_size', 50, 200)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    
    X, y = [], []
    for i in range(len(scaled_data) - int(window_size)):
        X.append(scaled_data[i:i + int(window_size)])
        y.append(scaled_data[i + int(window_size)])
    X, y = np.array(X), np.array(y)
    
    model = create_model(trial, window_size, loss_functions)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    
    history = model.fit(X, y, epochs=25, batch_size=int(batch_size), 
                        validation_split=trial.suggest_categorical('validation_split', [0.2, 0.3, 0.4]), 
                        callbacks=[early_stopping, reduce_lr], 
                        verbose=0)
    mse = history.history['val_mean_absolute_error'][-1]
    mse = history.history['val_mean_squared_error'][-1]
    return mse,mae

def new_lstm(ti, data, cmp):
    script_name = ti
    study_name = script_name + '1_study'
    storage = 'sqlite:///' + script_name + '_study.db'

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)) 
    sampler = TPESampler()   
    study = create_study(directions=['minimize', 'minimize'], study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler)
    study.optimize(lambda trial: optimize_model(trial, scaled_data), n_trials=100, n_jobs=8)
    best_trials = study.best_trials
    best_trial = best_trials[0]  
    print("best_trial.params:", best_trial.params) 
    best_model = create_model(best_trial, int(best_trial.params['window_size']), loss_functions)
    print("best_model.summary:", best_model.summary()) 
    X, y = [], []
    for i in range(len(scaled_data) - int(best_trial.params['window_size'])):
      X.append(scaled_data[i:i + int(best_trial.params['window_size'])])
      y.append(scaled_data[i + int(best_trial.params['window_size'])])
    X, y = np.array(X), np.array(y)
    best_model.fit(X, y, epochs=100, batch_size=int(best_trial.params['batch_size']), callbacks=[early_stopping], verbose=0)
    forecast_period = 126
    forecasted_prices = []
    window_size = int(best_trial.params['window_size'])
    current_data = scaled_data[-window_size:]
    forecasted_prices = []
    current_data = scaled_data[-window_size:]
    for _ in range(forecast_period):
      current_data_reshaped = current_data.reshape(1, window_size, 1)
      prediction = best_model.predict(current_data_reshaped)
      forecasted_price = prediction[0, 0]  
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
  data = stk_dt(t[0])
  cmp = data.iloc[-1].round(2)
  t[22], t[23], t[24], t[25] = new_lstm(t[0], data,cmp)
  t[31] = 'Y'
  print("Forecasted prices:", t[22], t[23], t[24], t[25])
  with open('upddata.txt', 'a') as f:
    f.write(str(t) + '\n')
print(a)
print(b)
os.environ['a'] = str(a)
os.environ['b'] = str(b)
