import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_input(data):
    """
    Preprocess the input data for the LSTM model.
    
    Parameters:
    - data: A DataFrame containing the input features.
    
    Returns:
    - processed_data: A numpy array suitable for LSTM input.
    """
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    sequence_length = 10  # Example sequence length
    X = []
    
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
    
    return np.array(X), scaler

def inverse_transform(scaler, data):
    """
    Inverse transform the scaled data back to original values.
    
    Parameters:
    - scaler: The fitted MinMaxScaler.
    - data: The scaled data to inverse transform.
    
    Returns:
    - original_data: The data in its original scale.
    """
    return scaler.inverse_transform(data)