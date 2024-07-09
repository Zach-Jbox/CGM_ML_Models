from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os
import time
from config import DATABASE_PATH
from database import fetch_data_for_model_training, save_prediction, fetch_data_for_model_training_lstm
from utils import generate_graph

def create_dataset(X, y, time_steps=1, future_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - future_steps + 1):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps + future_steps - 1])
    return np.array(Xs), np.array(ys)

def fetch_data_for_model_training_lstm():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 288", conn)
    conn.close()
    df = df.iloc[::-1].reset_index(drop=True)  # Reverse to keep the chronological order
    return df

def update_rf_predictions():
    while True:
        df = fetch_data_for_model_training()

        if df.empty:
            time.sleep(300)
            continue

        df['hour'] = df['hour'].astype(int)
        df['minute'] = df['minute'].astype(int)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(df[['hour', 'minute']], df['glucose_level'])
        joblib.dump(rf_model, 'random_forest_model.pkl')

        next_hour = (df['hour'].iloc[-1] + (df['minute'].iloc[-1] + 30) // 60) % 24
        next_minute = (df['minute'].iloc[-1] + 30) % 60
        next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])
        prediction = rf_model.predict(next_data_point)
        rounded_prediction = int(round(prediction[0]))
        save_prediction("RF_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        time.sleep(300)

def update_xgb_predictions():
    while True:
        df = fetch_data_for_model_training()

        if df.empty:
            time.sleep(300)
            continue

        df['hour'] = df['hour'].astype(int)
        df['minute'] = df['minute'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(df[['hour', 'minute']], df['glucose_level'], test_size=0.2, random_state=42)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, 'xgboost_model.pkl')

        next_hour = (df['hour'].iloc[-1] + (df['minute'].iloc[-1] + 30) // 60) % 24
        next_minute = (df['minute'].iloc[-1] + 30) % 60
        next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])
        prediction = model.predict(next_data_point)
        rounded_prediction = int(round(prediction[0]))
        save_prediction("XGB_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        time.sleep(300)

def update_lstm_predictions():
    while True:
        df = fetch_data_for_model_training_lstm()

        if df.empty:
            time.sleep(300)
            continue

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['glucose_level']])
        time_steps = 10
        future_steps_2hours = 24

        X_2hours, y_2hours = create_dataset(data_scaled, data_scaled, time_steps, future_steps_2hours)
        if X_2hours.size == 0 or y_2hours.size == 0:
            time.sleep(300)
            continue

        model_path = 'lstm_model_2hours.h5'
        if not os.path.exists(model_path):
            create_and_save_lstm_model()

        model_2hours = load_model(model_path)

        X_pred_2hours = np.array([data_scaled[-time_steps:]])
        y_pred_2hours = model_2hours.predict(X_pred_2hours)
        y_pred_rescaled_2hours = scaler.inverse_transform(y_pred_2hours.reshape(-1, 1))
        rounded_prediction = int(round(y_pred_rescaled_2hours[0][0]))

        last_hour = df['hour'].iloc[-1]
        last_minute = df['minute'].iloc[-1]
        next_hour = (last_hour + (last_minute + 120) // 60) % 24
        next_minute = (last_minute + 120) % 60

        save_prediction("LSTM_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        time.sleep(300)

def create_and_save_lstm_model():
    df = fetch_data_for_model_training_lstm()
    if df.empty:
        return

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['glucose_level']])
    time_steps = 10
    future_steps_2hours = 24

    X_2hours, y_2hours = create_dataset(data_scaled, data_scaled, time_steps, future_steps_2hours)
    if X_2hours.size == 0 or y_2hours.size == 0:
        return

    X_train_2hours, X_test_2hours, y_train_2hours, y_test_2hours = train_test_split(X_2hours, y_2hours, test_size=0.2, shuffle=False)
    model_2hours = Sequential()
    model_2hours.add(LSTM(units=64, input_shape=(X_train_2hours.shape[1], X_train_2hours.shape[2])))
    model_2hours.add(Dense(units=1))
    model_2hours.compile(optimizer='adam', loss='mean_squared_error')
    model_2hours.fit(X_train_2hours, y_train_2hours, epochs=100, batch_size=32, validation_split=0.2)
    model_2hours.save('lstm_model_2hours.h5')

def update_graphs():
    while True:
        conn = sqlite3.connect(DATABASE_PATH)

        # Fetch actual data
        df_actual = pd.read_sql_query("SELECT glucose_level FROM GLUCOSE_READINGS ORDER BY id", conn)

        # Fetch predictions
        rf_pred = pd.read_sql_query("SELECT prediction FROM RF_PREDICTIONS ORDER BY id", conn)
        xgb_pred = pd.read_sql_query("SELECT prediction FROM XGB_PREDICTIONS ORDER BY id", conn)
        lstm_pred = pd.read_sql_query("SELECT prediction FROM LSTM_PREDICTIONS ORDER BY id", conn)

        conn.close()

        if df_actual.empty or rf_pred.empty or xgb_pred.empty or lstm_pred.empty:
            time.sleep(300)
            continue

        # Convert data to lists for plotting
        actual_data = df_actual['glucose_level'].tolist()
        rf_predictions = rf_pred['prediction'].tolist()
        xgb_predictions = xgb_pred['prediction'].tolist()
        lstm_predictions = lstm_pred['prediction'].tolist()

        # Generate and save the graphs
        generate_graph('rf', actual_data, rf_predictions, 'rf_predictions_vs_actual.png')
        generate_graph('xgb', actual_data, xgb_predictions, 'xgb_predictions_vs_actual.png')
        generate_graph('lstm', actual_data, lstm_predictions, 'lstm_predictions_vs_actual.png')

        time.sleep(300)  # Update every 5 minutes