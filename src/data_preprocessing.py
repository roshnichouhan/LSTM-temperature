import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_data(data_path, lookback):

    # Load CSV
    df = pd.read_csv(data_path)

    # Get temperature column
    temperature = df["temperature"].values.reshape(-1, 1)

    # Scale values between 0 and 1
    scaler = MinMaxScaler()
    temperature_scaled = scaler.fit_transform(temperature)

    # Create sequences
    X = []
    y = []

    for i in range(len(temperature_scaled) - lookback):
        X.append(temperature_scaled[i:i + lookback])
        y.append(temperature_scaled[i + lookback])

    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler