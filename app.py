import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/temperature_lstm.keras"
DATA_PATH = "data/temperature.csv"
LOOKBACK = 5


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LSTM Temperature Forecasting",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #ffffff;
        text-align: center;
    }

    .metric-title {
        font-size: 15px;
        color: #666;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_temperature_data():
    return pd.read_csv(DATA_PATH)


# =========================================================
# APPLICATION HEADER
# =========================================================

st.markdown(
    '<div class="main-title">LSTM Temperature Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Time-series forecasting application using an LSTM neural network
    to predict the next temperature from historical observations.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** LSTM Neural Network")
    st.write("**Task:** Time-Series Forecasting")
    st.write("**Lookback Window:** 5 observations")
    st.write("**Optimizer:** Adam")
    st.write("**Loss Function:** Mean Squared Error")

    st.divider()

    st.subheader("Pipeline")

    st.write("1. Historical temperature data")
    st.write("2. Min-Max normalization")
    st.write("3. Sequence creation")
    st.write("4. LSTM feature learning")
    st.write("5. Dense regression layer")
    st.write("6. Next-temperature prediction")


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

try:

    model = load_lstm_model()
    df = load_temperature_data()

except Exception as e:

    st.error(
        "Unable to load the trained model or temperature dataset."
    )

    st.exception(e)

    st.stop()


# =========================================================
# VALIDATE DATA
# =========================================================

if "temperature" not in df.columns:

    st.error(
        "The dataset must contain a 'temperature' column."
    )

    st.stop()


# =========================================================
# CREATE SCALER
# =========================================================

scaler = MinMaxScaler()

historical = df["temperature"].values.reshape(-1, 1)

scaler.fit(historical)


# =========================================================
# DATASET SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Observations",
        len(df)
    )

with col2:
    st.metric(
        "Minimum",
        f"{df['temperature'].min():.2f} °C"
    )

with col3:
    st.metric(
        "Maximum",
        f"{df['temperature'].max():.2f} °C"
    )

with col4:
    st.metric(
        "Average",
        f"{df['temperature'].mean():.2f} °C"
    )


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">Input Historical Temperatures</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the five most recent temperature observations. "
    "The LSTM model will use this sequence to forecast the next value."
)


temperatures = []

input_columns = st.columns(LOOKBACK)

for i, column in enumerate(input_columns):

    with column:

        value = st.number_input(
            f"Observation {i + 1}",
            min_value=-50.0,
            max_value=60.0,
            value=25.0 + i,
            step=0.1
        )

        temperatures.append(value)


# =========================================================
# PREDICTION
# =========================================================

st.write("")

predict_button = st.button(
    "Forecast Next Temperature",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Convert input to NumPy array
    data = np.array(
        temperatures
    ).reshape(-1, 1)

    # Normalize using historical-data scaler
    scaled_data = scaler.transform(data)

    # Reshape for LSTM
    X = scaled_data.reshape(
        1,
        LOOKBACK,
        1
    )

    # Make prediction
    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    # Convert prediction back to Celsius
    prediction = scaler.inverse_transform(
        prediction_scaled
    )

    predicted_temperature = float(
        prediction[0][0]
    )

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">Forecast Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:

        st.success(
            f"Predicted Next Temperature: "
            f"{predicted_temperature:.2f} °C"
        )

    with result_col2:

        st.metric(
            "Prediction",
            f"{predicted_temperature:.2f} °C"
        )

    # =====================================================
    # INPUT VISUALIZATION
    # =====================================================

    st.markdown(
        '<div class="section-title">Temperature Sequence</div>',
        unsafe_allow_html=True
    )

    sequence_df = pd.DataFrame(
        {
            "Observation": range(
                1,
                LOOKBACK + 1
            ),
            "Temperature": temperatures
        }
    )

    fig = plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        sequence_df["Observation"],
        sequence_df["Temperature"],
        marker="o"
    )

    plt.axhline(
        predicted_temperature,
        linestyle="--",
        label="Predicted Next Temperature"
    )

    plt.xlabel("Observation")
    plt.ylabel("Temperature (°C)")
    plt.title("Historical Temperature Sequence")

    plt.legend()

    plt.tight_layout()

    st.pyplot(fig)



# =========================================================


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">Project Highlights</div>',
    unsafe_allow_html=True
)

highlight_col1, highlight_col2, highlight_col3 = st.columns(3)

with highlight_col1:

    st.write("### LSTM")

    st.write(
        "Captures sequential and temporal dependencies "
        "in historical temperature data."
    )


with highlight_col2:

    st.write("### Data Scaling")

    st.write(
        "MinMaxScaler normalizes the input before "
        "feeding it into the neural network."
    )


with highlight_col3:

    st.write("### Regression")

    st.write(
        "The final Dense layer produces a continuous "
        "temperature prediction."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "LSTM Temperature Forecasting | "
    "TensorFlow • Python • Streamlit • Scikit-learn"
)