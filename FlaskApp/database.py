import sqlite3
from datetime import datetime
from config import DATABASE_PATH
import pandas as pd

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS GLUCOSE_READINGS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            glucose_level REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RF_PREDICTIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            prediction REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS XGB_PREDICTIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            prediction REAL NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LSTM_PREDICTIONS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            prediction REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def trim_table(table_name, max_rows=864):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    if count > max_rows:
        cursor.execute(f"DELETE FROM {table_name} WHERE id IN (SELECT id FROM {table_name} ORDER BY id ASC LIMIT ?)", (count - max_rows,))
    conn.commit()
    conn.close()

def add_glucose_reading(glucose_value):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute("INSERT INTO GLUCOSE_READINGS (hour, minute, glucose_level) VALUES (?, ?, ?)",
                   (now.hour, now.minute, glucose_value))
    conn.commit()
    conn.close()
    print(f"Added glucose reading: hour={now.hour}, minute={now.minute}, glucose_level={glucose_value}")
    trim_table('GLUCOSE_READINGS')

def save_prediction(table_name, hour, minute, prediction):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {table_name} (hour, minute, prediction) VALUES (?, ?, ?)",
                   (int(hour), int(minute), float(prediction)))
    conn.commit()
    conn.close()
    print(f"Saved prediction: table={table_name}, hour={hour}, minute={minute}, prediction={prediction}")
    trim_table(table_name)

def fetch_data_for_model_training():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM GLUCOSE_READINGS ORDER BY id DESC LIMIT 864", conn)
    conn.close()
    df = df.iloc[::-1].reset_index(drop=True)  # Reverse to keep the chronological order
    return df