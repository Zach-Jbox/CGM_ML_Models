import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def generate_graph(model, actual_data, predicted_data, file_path):
    actual_data = actual_data[-288:]
    predicted_data = predicted_data[-288:]

    # Create a time index for the last 24 hours
    time_index = pd.date_range(end=pd.Timestamp.now(), periods=288, freq='5T')
    
    # Select labels for every 2 hours
    label_indices = range(0, len(time_index), 24)  # 24 * 5 minutes = 120 minutes (2 hours)
    labels = [time_index[i].strftime('%H:%M') for i in label_indices]
    positions = [i for i in label_indices]

    plt.figure()
    plt.plot(range(len(actual_data)), actual_data, label='Actual Data')
    plt.plot(range(len(predicted_data)), predicted_data, label='Predicted Data')
    plt.xlabel('Time')
    plt.ylabel('Glucose Level')
    plt.title(f'{model.upper()} Predictions vs Actual')
    plt.xticks(ticks=positions, labels=labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()