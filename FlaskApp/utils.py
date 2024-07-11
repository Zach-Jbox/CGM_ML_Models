import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def generate_graph(model, actual_data, predicted_data, file_path, offset=0):
    # Convert lists to pandas Series if necessary
    if isinstance(actual_data, list):
        actual_data = pd.Series(actual_data)
    if isinstance(predicted_data, list):
        predicted_data = pd.Series(predicted_data)

    # Ensure data is only the last 288 points
    actual_data = actual_data[-288:].reset_index(drop=True)
    predicted_data = predicted_data[-(288 + offset):-offset if offset != 0 else None].reset_index(drop=True)

    # Generate simple indices for the x-axis
    indices = range(len(actual_data))

    plt.figure(figsize=(10, 6))
    plt.plot(indices, actual_data, label='Actual Data', linestyle='-', color='blue')
    plt.plot(indices, predicted_data, label='Predicted Data', linestyle='-', color='orange')
    plt.scatter(indices, actual_data, color='blue', s=10)  # Add dots to actual data points
    plt.scatter(indices, predicted_data, color='orange', s=10)  # Add dots to predicted data points
    plt.xlabel('Data Points')
    plt.ylabel('Glucose Level')
    plt.title(f'{model.upper()} Predictions vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()