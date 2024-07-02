import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_accuracy_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = calculate_mape(actual, predicted)
    
    return {
        "MAE": round(mae, 1),
        "MSE": round(mse, 1),
        "RMSE": round(rmse, 1),
        "MAPE": round(mape, 1)
    }