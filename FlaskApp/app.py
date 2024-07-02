from flask import Flask, jsonify, send_file
import pandas as pd
import sqlite3
import os
from database import init_db, DATABASE_PATH
from tasks import start_background_tasks
from clarke_error_grid_analysis import clarke_error_grid
from metrics import calculate_accuracy_metrics

app = Flask(__name__)

init_db()
start_background_tasks()

@app.route('/metrics', methods=['GET'])
def get_all_metrics():
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Fetch data for Random Forest
    df_rf = pd.read_sql_query("SELECT glucose_level FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 288", conn)
    pred_df_rf = pd.read_sql_query("SELECT prediction FROM RF_PREDICTIONS ORDER BY id DESC LIMIT 288", conn)
    
    # Fetch data for XGBoost
    df_xgb = pd.read_sql_query("SELECT glucose_level FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 288", conn)
    pred_df_xgb = pd.read_sql_query("SELECT prediction FROM XGB_PREDICTIONS ORDER BY id DESC LIMIT 288", conn)
    
    # Fetch data for LSTM
    df_lstm = pd.read_sql_query("SELECT glucose_level FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 288", conn)
    pred_df_lstm = pd.read_sql_query("SELECT prediction FROM LSTM_PREDICTIONS ORDER BY id DESC LIMIT 288", conn)
    
    conn.close()
    
    if df_rf.empty or pred_df_rf.empty or df_xgb.empty or pred_df_xgb.empty or df_lstm.empty or pred_df_lstm.empty:
        return jsonify({'error': 'Not enough data for accuracy metrics'})

    rf_metrics = calculate_accuracy_metrics(df_rf['glucose_level'], pred_df_rf['prediction'])
    xgb_metrics = calculate_accuracy_metrics(df_xgb['glucose_level'], pred_df_xgb['prediction'])
    lstm_metrics = calculate_accuracy_metrics(df_lstm['glucose_level'], pred_df_lstm['prediction'])

    return jsonify({
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics,
        'LSTM': lstm_metrics
    })

@app.route('/predict_random_forest', methods=['GET'])
def predict_random_forest():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM RF_PREDICTIONS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No predictions available'})

    latest_prediction = df.iloc[0]
    return jsonify({'predicted_glucose': latest_prediction['prediction']})

@app.route('/predict_xgboost', methods=['GET'])
def predict_xgboost():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM XGB_PREDICTIONS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No predictions available'})

    latest_prediction = df.iloc[0]
    return jsonify({'predicted_glucose': latest_prediction['prediction']})

@app.route('/predict_lstm', methods=['GET'])
def predict_lstm():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM LSTM_PREDICTIONS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No predictions available'})

    latest_prediction = df.iloc[0]
    return jsonify({'predicted_glucose_2hours': latest_prediction['prediction']})

@app.route('/clarke_error_grid/rf', methods=['GET'])
def clarke_error_grid_rf():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT glucose_level FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 288", conn)
    pred_df = pd.read_sql_query("SELECT prediction FROM RF_PREDICTIONS ORDER BY id DESC LIMIT 288", conn)
    conn.close()

    if df.empty or pred_df.empty:
        return jsonify({'error': 'Not enough data for Clarke Error Grid analysis'})

    plt, zone_counts = clarke_error_grid(df['glucose_level'], pred_df['prediction'], "Random Forest")
    plt.savefig('clarke_error_grid_rf.png')
    plt.close()

    return send_file('clarke_error_grid_rf.png', mimetype='image/png')

@app.route('/clarke_error_grid/xgb', methods=['GET'])
def clarke_error_grid_xgb():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT glucose_level FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 288", conn)
    pred_df = pd.read_sql_query("SELECT prediction FROM XGB_PREDICTIONS ORDER BY id DESC LIMIT 288", conn)
    conn.close()

    if df.empty or pred_df.empty:
        return jsonify({'error': 'Not enough data for Clarke Error Grid analysis'})

    plt, zone_counts = clarke_error_grid(df['glucose_level'], pred_df['prediction'], "XGBoost")
    plt.savefig('clarke_error_grid_xgb.png')
    plt.close()

    return send_file('clarke_error_grid_xgb.png', mimetype='image/png')

@app.route('/graph/<model>', methods=['GET'])
def get_graph(model):
    valid_models = ['rf', 'xgb', 'lstm']
    if model in valid_models:
        file_path = f'{model}_predictions_vs_actual.png'
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Graph not found'}), 404
    else:
        return jsonify({'error': 'Invalid model'}), 400

@app.route('/current_glucose', methods=['GET'])
def current_glucose():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({'error': 'No glucose readings available'})

    latest_reading = df.iloc[0]
    glucose_level = latest_reading['glucose_level']
    status = 'Normal'

    if glucose_level < 75:
        status = 'Low'
    elif glucose_level > 150:
        status = 'High'

    return jsonify({'glucose_level': glucose_level, 'status': status})

if __name__ == "__main__":
    app.run(debug=False, port=5000)