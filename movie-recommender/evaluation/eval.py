import numpy as np

# Defined a function to calculate Root Mean Square Error (RMSE)
def rmse(y_true, y_pred):
    # Converted true and predicted values into NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Computed RMSE to measure prediction accuracy
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
