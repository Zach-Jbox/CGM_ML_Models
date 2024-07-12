import threading
import time
from pydexcom import Dexcom
from datetime import datetime
from database import add_glucose_reading
from models import update_rf_predictions, update_xgb_predictions, update_lstm_predictions, update_graphs

global current_trend_arrow
current_trend_arrow = None

def update_glucose_readings():
    global current_trend_arrow  # Ensure that we are referring to the global variable
    dexcom = Dexcom("Username", "Password")
    while True:
        try:
            glucose_reading = dexcom.get_current_glucose_reading()
            add_glucose_reading(glucose_reading.value)
            current_trend_arrow = glucose_reading.trend_arrow
            print(f"Reading added: {glucose_reading.value} {glucose_reading.trend_arrow} at {datetime.now()}")
            print(f"Current trend arrow updated to: {current_trend_arrow}")
        except Exception as e:
            print(f"Error updating glucose reading: {e}")
        time.sleep(300)  # Sleep for 5 minutes before the next reading
        
def start_background_tasks():
    rf_thread = threading.Thread(target=update_rf_predictions)
    xgb_thread = threading.Thread(target=update_xgb_predictions)
    lstm_thread = threading.Thread(target=update_lstm_predictions)
    update_thread = threading.Thread(target=update_glucose_readings)
    graph_thread = threading.Thread(target=update_graphs)

    rf_thread.daemon = True
    xgb_thread.daemon = True
    lstm_thread.daemon = True
    update_thread.daemon = True
    graph_thread.daemon = True

    rf_thread.start()
    xgb_thread.start()
    lstm_thread.start()
    update_thread.start()
    graph_thread.start()