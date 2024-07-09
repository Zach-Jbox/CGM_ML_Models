import sqlite3
import pandas as pd
from utils import generate_graph  # Ensure this imports the generate_graph function from utils.py

DATABASE_PATH = 'glucose.db'

def fetch_last_n_entries(table_name, n):
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT {n}", conn)
    df = df.iloc[::-1]  # Reverse to get chronological order
    conn.close()
    return df

# Fetch the data
actual_data_df = fetch_last_n_entries("GLUCOSE_READINGS", 312)  # 288 + 24 for LSTM offset
actual_data = actual_data_df['glucose_level']

rf_predictions_df = fetch_last_n_entries("RF_PREDICTIONS", 288)
rf_predictions = rf_predictions_df['prediction']

xgb_predictions_df = fetch_last_n_entries("XGB_PREDICTIONS", 288)
xgb_predictions = xgb_predictions_df['prediction']

lstm_predictions_df = fetch_last_n_entries("LSTM_PREDICTIONS", 288)
lstm_predictions = lstm_predictions_df['prediction']

# Generate the graphs with appropriate offsets
generate_graph('rf', actual_data[-288:], rf_predictions, 'rf_predictions_vs_actual.png', offset=6)
generate_graph('xgb', actual_data[-288:], xgb_predictions, 'xgb_predictions_vs_actual.png', offset=6)
generate_graph('lstm', actual_data[-312:], lstm_predictions, 'lstm_predictions_vs_actual.png', offset=24)