import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, GRU
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError,MeanAbsolutePercentageError , MeanSquaredLogarithmicError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import TimeSeriesSplit

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

tickers = ['TCS.NS','INFY.NS']
early_stopping = EarlyStopping(monitor='mean_squared_error', patience=5 )

def stk_dt(tk,scaler):
   data = yf.download(tk, period='5y')['Close'].dropna()
   last_date = pd.to_datetime(data.index[-1].to_pydatetime().date())
   scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
   return scaled_data,last_date


@tf.function(experimental_reduce_retracing=True)
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    msle = tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred)
    return 0.6 * mse + 0.3 * mae + 0.1 * msle
  
def create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(int(lstm_units), return_sequences=True, input_shape=(window_size, 1))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(GRU(int(gru_units), return_sequences=True)))
    model.add(Dense(1))
    
    #def combined_loss(y_true, y_pred):
     #   mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
      #  mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
      #  msle = tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred)
       # return 0.6 * mse + 0.3 * mae + 0.1 * msle
          
    model.compile(
        optimizer=['adam', 'rmsprop', 'sgd'][int(optimizer_idx)],
        loss=combined_loss,
        metrics=[
            'mean_absolute_error', 
            'mean_squared_error', 
            'mean_absolute_percentage_error', 
            'mean_squared_logarithmic_error'        ])
    return model
    
def optimize_model(trial,scaled_data):
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    gru_units = trial.suggest_int('gru_units', 20, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 32, 64)
    optimizer_idx = trial.suggest_int('optimizer_idx', 0, 2)
    window_size = trial.suggest_int('window_size', 50, 250)

    X, y = [], []
    for i in range(len(scaled_data) - int(window_size)):
        X.append(scaled_data[i:i + int(window_size)])
        y.append(scaled_data[i + int(window_size)])
    X, y = np.array(X), np.array(y)  
    model = create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size,window_size)
    history = model.fit(X, y, epochs=50, batch_size=int(batch_size), validation_split=0.2, callbacks=[early_stopping], verbose=0)

    mae = history.history['val_mean_absolute_error'][-1]
    mse = history.history['val_mean_squared_error'][-1]
    mape = history.history['val_mean_absolute_percentage_error'][-1]
    #mase = history.history['val_mean_absolute_scaled_error'][-1]
    rmse =  np.sqrt(mse)
    msle = history.history['val_mean_squared_logarithmic_error'][-1]
    return mae, mse, rmse, msle,mape
   
def new_lstm(ti, scaled_data, scaler,lst):
    for filename in os.listdir():
        if filename.endswith('_study.db'):
            os.remove(filename)
    script_name= ti
    study_name = script_name + '_study'
    storage = 'sqlite:///' + script_name + '_study.db'
    
    study = optuna.create_study(directions=['minimize', 'minimize','minimize','minimize', 'minimize'], study_name=study_name, storage=storage, load_if_exists=True, sampler=TPESampler())
    study.optimize(lambda trial: optimize_model(trial, scaled_data), n_trials=25, n_jobs=8)
    best_trials = study.best_trials
    best_trial = best_trials[0]  # Select the first best trial
    best_model = create_model(**best_trial.params)
    X, y = [], []
    for i in range(len(scaled_data) - int(best_trial.params['window_size'])):
      X.append(scaled_data[i:i + int(best_trial.params['window_size'])])
      y.append(scaled_data[i + int(best_trial.params['window_size'])])
    X, y = np.array(X), np.array(y)
    best_model.fit(X, y, epochs=100, batch_size=int(best_trial.params['batch_size']),callbacks=[early_stopping], verbose=0)
    last_date = lst
    forecast_dates = pd.date_range(start=last_date, periods=126, freq='D')
    forecasted_prices = []
    current_data = scaled_data[-int(best_trial.params['window_size']):]
    for date in forecast_dates:
        prediction = best_model.predict(current_data.reshape(1, int(best_trial.params['window_size']), 1))[:, -1, :]
        forecasted_prices.append(prediction[0, 0])
        current_data = np.append(current_data[1:], prediction[0, 0])
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))
    print("forecasted_prices:",forecasted_prices)
    return forecasted_prices

for t in tickers:
    scaler = MinMaxScaler()
    scaled_data,lst = stk_dt(t,scaler)
    f = new_lstm(t, scaled_data, scaler,lst)
    print("Stock name:", t)
    print("Forecasted prices:", f)
