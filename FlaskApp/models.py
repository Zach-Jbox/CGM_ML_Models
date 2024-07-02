from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import numpy as np
import os
import json
import sqlite3
import time
from config import MODEL_PATHS
from config import DATABASE_PATH
from database import fetch_data_for_model_training, save_prediction
from utils import generate_graph

def create_dataset(X, y, time_steps=1, future_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - future_steps + 1):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps + future_steps - 1])
    return np.array(Xs), np.array(ys)

def update_rf_predictions():
    while True:
        df = fetch_data_for_model_training()

        if df.empty:
            print("No data available for Random Forest prediction")
            time.sleep(300)
            continue

        df['hour'] = df['hour'].astype(int)
        df['minute'] = df['minute'].astype(int)

        model_path = MODEL_PATHS['random_forest']
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(df[['hour', 'minute']], df['glucose_level'])
        joblib.dump(rf_model, model_path)

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
            print("No data available for XGBoost prediction")
            time.sleep(300)
            continue

        df['hour'] = df['hour'].astype(int)
        df['minute'] = df['minute'].astype(int)

        model_path = MODEL_PATHS['xgboost']
        X_train, X_test, y_train, y_test = train_test_split(df[['hour', 'minute']], df['glucose_level'], test_size=0.2, random_state=42)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

        next_hour = (df['hour'].iloc[-1] + (df['minute'].iloc[-1] + 30) // 60) % 24
        next_minute = (df['minute'].iloc[-1] + 30) % 60
        next_data_point = pd.DataFrame([[next_hour, next_minute]], columns=['hour', 'minute'])
        prediction = model.predict(next_data_point)
        rounded_prediction = int(round(prediction[0]))
        save_prediction("XGB_PREDICTIONS", next_hour, next_minute, rounded_prediction)
        time.sleep(300)

def update_lstm_predictions():
    while True:
        df = fetch_data_for_model_training()

        if df.empty:
            print("No data available for LSTM prediction")
            time.sleep(300)
            continue

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['glucose_level']])
        time_steps = 10
        future_steps_2hours = 24

        X_2hours, y_2hours = create_dataset(data_scaled, data_scaled, time_steps, future_steps_2hours)
        if X_2hours.size == 0 or y_2hours.size == 0:
            print("Not enough data to create the LSTM dataset")
            time.sleep(300)
            continue

        model_path = MODEL_PATHS['lstm']
        model_json_path = MODEL_PATHS['lstm_json']
        if not os.path.exists(model_path) or not os.path.exists(model_json_path):
            print(f"LSTM model file not found: {model_path}")
            create_and_save_lstm_model()

        model_2hours = load_lstm_model(model_path, model_json_path)

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
    df = fetch_data_for_model_training()
    if df.empty:
        print("No data available to create the LSTM model")
        return

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['glucose_level']])
    time_steps = 10
    future_steps_2hours = 24

    X_2hours, y_2hours = create_dataset(data_scaled, data_scaled, time_steps, future_steps_2hours)
    if X_2hours.size == 0 or y_2hours.size == 0:
        print("Not enough data to create the LSTM dataset")
        return

    model_path = MODEL_PATHS['lstm']
    model_json_path = MODEL_PATHS['lstm_json']
    X_train_2hours, X_test_2hours, y_train_2hours, y_test_2hours = train_test_split(X_2hours, y_2hours, test_size=0.2, shuffle=False)
    model_2hours = Sequential()
    model_2hours.add(LSTM(units=64, input_shape=(X_train_2hours.shape[1], X_train_2hours.shape[2])))
    model_2hours.add(Dense(units=1))
    model_2hours.compile(optimizer='adam', loss='mean_squared_error')
    model_2hours.fit(X_train_2hours, y_train_2hours, epochs=100, batch_size=32, validation_split=0.2)
    model_2hours.save(model_path)

    with open(model_json_path, 'w') as json_file:
        json_file.write(model_2hours.to_json())

def load_lstm_model(model_path, model_json_path):
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()

    model_2hours = model_from_json(model_json, custom_objects={'LSTM': LSTM})
    model_2hours.load_weights(model_path)
    return model_2hours

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
            print("No data available for generating graphs")
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