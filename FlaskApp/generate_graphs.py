import sqlite3
import pandas as pd
from utils import generate_graph  # Ensure this imports the generate_graph function from utils.py

DATABASE_PATH = 'glucose.db'

def fetch_last_288_entries(table_name):
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 288", conn)
    df = df.iloc[::-1]  # Reverse to get chronological order
    conn.close()
    return df

# Fetch the data
actual_data_df = fetch_last_288_entries("GLUCOSE_READINGS")
actual_data = actual_data_df['glucose_level']

rf_predictions_df = fetch_last_288_entries("RF_PREDICTIONS")
rf_predictions = rf_predictions_df['prediction']

xgb_predictions_df = fetch_last_288_entries("XGB_PREDICTIONS")
xgb_predictions = xgb_predictions_df['prediction']

lstm_predictions_df = fetch_last_288_entries("LSTM_PREDICTIONS")
lstm_predictions = lstm_predictions_df['prediction']

# Generate the graphs
generate_graph('rf', actual_data, rf_predictions, 'rf_predictions_vs_actual.png')
generate_graph('xgb', actual_data, xgb_predictions, 'xgb_predictions_vs_actual.png')
generate_graph('lstm', actual_data, lstm_predictions, 'lstm_predictions_vs_actual.png')