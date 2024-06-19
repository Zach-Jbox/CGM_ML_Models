from flask import Flask, jsonify
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import time
import threading
from pydexcom import Dexcom

app = Flask(__name__)

# Connect to the database and create the GLUCOSE_READINGS table
conn = sqlite3.connect('glucose.db')
cursor = conn.cursor()

glucose_readings = """CREATE TABLE IF NOT EXISTS GLUCOSE_READINGS(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    glucose_level REAL NOT NULL
);"""
cursor.execute(glucose_readings)

conn.commit()
conn.close()

def add_glucose_reading(glucose_value):
    conn = sqlite3.connect('glucose.db')
    cursor = conn.cursor()

    now = datetime.now()
    hour = now.hour
    minute = now.minute

    cursor.execute("INSERT INTO GLUCOSE_READINGS (hour, minute, glucose_level) VALUES (?, ?, ?)",
                   (hour, minute, glucose_value))
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM GLUCOSE_READINGS")
    total_entries = cursor.fetchone()[0]

    if total_entries > 288:
        cursor.execute("DELETE FROM GLUCOSE_READINGS WHERE id IN (SELECT id FROM GLUCOSE_READINGS ORDER BY id LIMIT ?)", 
                       (total_entries - 288,))
        conn.commit()

    conn.close()

def update_glucose_readings():
    dexcom = Dexcom("Username", "Password")
    while True:
        glucose_reading = dexcom.get_current_glucose_reading()
        add_glucose_reading(glucose_reading.value)
        print(f"Reading added: {glucose_reading.value} at {datetime.now()}")
        time.sleep(300)

# Start the Dexcom update in a separate thread
update_thread = threading.Thread(target=update_glucose_readings)
update_thread.start()

# Define the route for the prediction
@app.route('/', methods=['GET'])
def predict_glucose():
    # Fetch data from the database
    conn = sqlite3.connect('glucose.db')
    query = "SELECT * FROM GLUCOSE_READINGS"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert columns to appropriate data types
    df['hour'] = df['hour'].astype(int)
    df['minute'] = df['minute'].astype(int)
    df['glucose_level'] = df['glucose_level'].astype(float)

    # Prepare data for prediction
    next_hour = df['hour'].iloc[-1]
    next_minute = df['minute'].iloc[-1] + 30

    if next_minute >= 60:
        next_hour += 1
        next_minute -= 60

    next_data_point = [[next_hour, next_minute]]

    # Initialize and train the Random Forest Regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(df[['hour', 'minute']], df['glucose_level'])

    # Make prediction
    next_prediction = rf_model.predict(next_data_point)

    # Return the prediction as JSON
    return jsonify({'predicted_glucose': next_prediction[0]})

if __name__=="__main__":
    app.run(debug=False,port=5000)

#XGBoost

#LSTM