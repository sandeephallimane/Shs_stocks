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
from optuna.samplers import TPESampler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import TimeSeriesSplit

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
        tscv = TimeSeriesSplit(n_splits=5)

        cv_scores = []
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = create_model(lstm_units, gru_units, dropout_rate, optimizer_idx, batch_size, window_size)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            history = model.fit(X_train, y_train, epochs=50, batch_size=int(batch_size), validation_split=0.2, callbacks=[early_stopping], verbose=0)

            y_pred = model.predict(X_val)[:, -1, :]
            mae = np.mean(np.abs(y_val[:, -1, :] - y_pred[:, -1, :]))
            mse = np.mean((y_val[:, -1, :] - y_pred[:, -1, :]) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100 if y_val.all() != 0 else 100
            r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
            ic = np.corrcoef(y_val, y_pred)[0, 1]
            auc_roc = tf.keras.metrics.AUC(y_val, y_pred)
            msle = np.mean((np.log(y_val + 1) - np.log(y_pred + 1)) ** 2)
            mfe = np.mean(y_val - y_pred)
            mad = np.mean(np.abs(y_val - y_pred))
            aic = 2 * (len(model.layers) + 1) - 2 * np.log(np.mean(mse))

            cv_scores.append([mae, mse, rmse, mape, r2, ic, auc_roc, msle, mfe, mad, aic])

        avg_scores = np.mean(cv_scores, axis=0)
        return avg_scores

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
    n_jobs = 5
    study = optuna.create_study(directions=['minimize', 'minimize', 'minimize', 'minimize', 'maximize', 'maximize', 'maximize', 'minimize', 'minimize', 'minimize', 'minimize'],
                            study_name=study_name,
                            storage=storage,
                            load_if_exists=True,
                            sampler=TPESampler())
    study.optimize(optimize_model, n_trials=20, n_jobs=n_jobs)
    best_trials = study.best_trials
    best_trial = best_trials[0]
    best_model = create_model(**best_trial.params)
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    best_model.fit(train_data, epochs=50, batch_size=int(best_trial.params['batch_size']), validation_data=val_data, verbose=0)
    forecasted_prices = []
    current_data = test_data[-int(best_trial.params['window_size']):]
    for _ in range(126):
       prediction = best_model.predict(current_data.reshape(1, int(best_trial.params['window_size']), 1))[:, -1, :]
       forecasted_prices.append(prediction[0, 0])
       current_data = np.append(current_data[1:], prediction[0, 0])
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))
    return forecasted_prices

for t in tickers:
    f = new_lstm(t)
    print("Stock name:", t)
    print("Forecasted prices:", f)
