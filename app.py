import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


MODEL_PATH = "models/temperature_lstm.keras"

LOOKBACK = 5


st.set_page_config(
    page_title="LSTM Temperature Predictor",
    page_icon="🌡️",
    layout="centered"
)


st.title("🌡️ LSTM Temperature Predictor")

st.write(
    "This application uses an LSTM neural network "
    "to predict the next temperature from previous temperatures."
)


@st.cache_resource
def load_lstm_model():

    model = load_model(MODEL_PATH)

    return model


model = load_lstm_model()


st.subheader("Enter previous temperatures")

temperatures = []

for i in range(LOOKBACK):

    value = st.number_input(
        f"Day {i + 1} temperature (°C)",
        min_value=-50.0,
        max_value=60.0,
        value=25.0 + i,
        step=0.1
    )

    temperatures.append(value)


if st.button("Predict Next Temperature"):

    data = np.array(
        temperatures
    ).reshape(-1, 1)

    scaler = MinMaxScaler()

    # Fit scaler using historical training data
    df = pd.read_csv(
        "data/temperature.csv"
    )

    historical = df[
        "temperature"
    ].values.reshape(-1, 1)

    scaler.fit(historical)

    scaled_data = scaler.transform(data)

    X = scaled_data.reshape(
        1,
        LOOKBACK,
        1
    )

    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )

    predicted_temperature = float(
        prediction[0][0]
    )

    st.success(
        f"Predicted next temperature: "
        f"{predicted_temperature:.2f} °C"
    )


st.divider()

st.subheader("How does it work?")

st.code(
    """
Previous 5 temperatures
          ↓
     Scaling
          ↓
       LSTM
          ↓
    Dense Layer
          ↓
Next temperature
    """
)