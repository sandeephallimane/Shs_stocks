import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError,Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError,MeanAbsolutePercentageError , MeanSquaredLogarithmicError
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, AdamW, Nadam
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
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
from tensorflow.keras.models import save_model, load_model
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
filename2 = "newdata.txt"

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

def choose_scaler(data):
    # Shapiro-Wilk test
    stat, p = stats.shapiro(data)
    if p > 0.05:
        # Normally distributed, use standardization
        scaler = StandardScaler()
    else:
        # Not normally distributed, use normalization
        scaler = MinMaxScaler()
    return scaler
      
def stk_dt(tk):
   data1 = yf.download(tk, period='5y')['Close'].dropna()
   cmp = data1.iloc[-1].round(2)
   data = np.log(data1 / data1.shift(1)).dropna()
   z_score = (data - data.mean()) / data.std()
   data_without_outliers = data[(z_score < 2) & (z_score > -2)]
   print("data length:", len(data_without_outliers)) 
   return data_without_outliers, cmp

loss_functions = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanAbsoluteError()]
loss_categories = ['mse', 'mae']
loss_functions_dict = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError() 
    }

def create_model1(trial, window_size):
    loss = tf.keras.losses.MeanSquaredError()
    recurrent_dropout=0.2
    dropout=trial.suggest_float('dropout_rate', 0.2, 0.5)
    gru_unit=trial.suggest_categorical('gru_units', [50, 60,70,80,90,100])
    model = Sequential()
    model.add(LSTM(
        gru_unit,
        input_shape=(window_size, 1),  
        activation='tanh',
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    ))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=l2(0.2)))
    
    optimizers = [Adam(), RMSprop(), AdamW(), Nadam()]    
    model.compile(
       #optimizer=optimizers[trial.suggest_int('optimizer_idx', 0, 3)],
        optimizer= Nadam(),
        loss=loss,
        metrics=['mean_squared_error','mean_absolute_percentage_error']
    )
    
    return model
    
def create_model(trial, window_size, loss_functions):
    loss_name = trial.suggest_categorical('loss_function', loss_categories)
    loss = loss_functions_dict[loss_name]
    recurrent_dropout=trial.suggest_float('recurrent_dropout', 0.1, 0.2)
    dropout=trial.suggest_float('dropout_rate', 0.1, 0.4)
    activation=trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'swish', 'tanh'])
    kl=trial.suggest_float('l2', 0.01, 0.2)
    optimizers = [Adam(), RMSprop(), AdamW(), Nadam()]    
    ls =int(trial.suggest_int('lstm_units', 50, 150))
    
    model = Sequential()
    model.add(LSTM(
        ls, return_sequences=True,
        input_shape=(window_size, 1), 
        activation=activation,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(GRU(
        ls,return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    ))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=l2(kl))) 
    
    model.compile(
        optimizer=optimizers[trial.suggest_int('optimizer_idx', 0, 3)],
        loss=loss,
        metrics=['mean_squared_error', 'mean_absolute_percentage_error']
    )
    return model

    
early_stopping = EarlyStopping(monitor='mean_absolute_percentage_error', patience=15,restore_best_weights=True) 
    


def optimize_model(trial: Trial, scaled_data: np.ndarray):
    window_size = trial.suggest_categorical('window_size', [10, 20, 30, 40, 50, 60,70,80,90,100])
    batch_size = trial.suggest_categorical('batch_size', [32, 64]) 

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])
    X, y = np.array(X), np.array(y)    
    model = create_model(trial, window_size, loss_functions)
    class CustomEarlyStopping(tf.keras.callbacks.Callback):
      def __init__(self, min_epochs=30, patience=0, restore_best_weights=True):
        super(CustomEarlyStopping, self).__init__()
        self.min_epochs = min_epochs
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = float('inf')
        self.best_weights = None
        self.epochs_since_best = 0

      def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current is None:
            raise ValueError('`val_loss` is required in logs.')
        if epoch < self.min_epochs:
            return    
        if current < self.best:
            self.best = current
            self.epochs_since_best = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.epochs_since_best += 1  
        if self.epochs_since_best >= self.patience:
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None and epoch >= self.min_epochs:
                self.model.set_weights(self.best_weights)
                
    early_stopping = CustomEarlyStopping(min_epochs=30, patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=0.001)
    
    history = model.fit(X, y, epochs=60, batch_size=int(batch_size), validation_split=0.2, verbose=0)
    
    mape = history.history['val_mean_absolute_percentage_error'][-1]
    mse = history.history['val_mean_squared_error'][-1]
    model_filename = f"model_{trial.number}.keras"
    save_model(model, model_filename)
    
    return mape, mse

def new_lstm(ti, data, cmp):
    script_name = ti
    study_name = script_name + '21_study'
    storage = 'sqlite:///' + script_name + '_study.db'

    scaler = choose_scaler(data)
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)) 
    sampler = TPESampler()
    study = create_study(
        directions=['minimize', 'minimize'],
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=MedianPruner()  
    )
    study.optimize(lambda trial: optimize_model(trial, scaled_data), n_trials=200, n_jobs=8)
    
    best_trials = study.best_trials
    best_trial = best_trials[0]
    print("best_trial.params:", best_trial.params) 
    best_model_key = f'model_{best_trial.number}.keras'
    best_model = load_model(best_model_key)
    
    forecast_period = 124
    forecasted_prices = []
    window_size = int(best_trial.params['window_size'])
    current_data = scaled_data[-window_size:]
    ty=1
    for _ in range(forecast_period):
        current_data_reshaped = current_data.reshape(1, window_size, 1) 
        prediction = best_model.predict(current_data_reshaped)
        print("Day:",ty)
        print("Prediction:",prediction)
        forecasted_price = prediction[0,-1,0]  
        forecasted_prices.append(forecasted_price)
        current_data = np.append(current_data[1:], forecasted_price)
        ty=ty+1
    
    forecasted_price = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))
    fp=[]
    for f in forecasted_price:
        fp.append(cmp*(1+f)) 
    min_p = np.min(fp).round(2)
    max_p = np.max(fp).round(2)
    avg_p = np.mean(fp).round(2)
    ret_p = ((avg_p-cmp)*100/cmp).round(2)
    return min_p, max_p, avg_p, ret_p

b= download_file(url2, filename2)
if a>b:
  t = ast.literal_eval(lines[b])
  print("Stock name:", t[0])
  data,cmp= stk_dt(t[0])
  t[22], t[23], t[24], t[25] = new_lstm(t[0], data,cmp)
  t[31] = 'Y'
  print("Forecasted prices:", t[22], t[23], t[24], t[25])
  with open('newdata.txt', 'a') as f:
    f.write(str(t) + '\n')
print(a)
print(b)
os.environ['a'] = str(a)
os.environ['b'] = str(b)
