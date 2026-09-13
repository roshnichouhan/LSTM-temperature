import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    df = pd.read_csv(file_path)

    print("Dataset:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    return df


def create_sequences(data, lookback=5):
    X = []
    y = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def prepare_data(file_path, lookback=5):

    df = load_data(file_path)

    temperatures = df["temperature"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_temperature = scaler.fit_transform(temperatures)

    X, y = create_sequences(
        scaled_temperature,
        lookback
    )

    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )