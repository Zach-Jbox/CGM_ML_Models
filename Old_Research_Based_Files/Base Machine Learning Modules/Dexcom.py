from pydexcom import Dexcom
import time

dexcom = Dexcom("Username", "Password")

while True:
    glucose_reading = dexcom.get_current_glucose_reading()
    print(glucose_reading)
    time.sleep(300) 