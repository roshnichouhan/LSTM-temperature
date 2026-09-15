# 🌡️ LSTM Temperature Prediction

https://github.com/user-attachments/assets/233258aa-bf87-4bd0-87b1-94c2acad6f44

A deep learning project that uses a **Long Short-Term Memory (LSTM)** neural network to predict future temperature values from historical time-series data. The project includes data preprocessing, sequence generation, model training, evaluation, and an interactive **Streamlit dashboard** for making predictions and visualizing temperature trends.

## 🚀 Project Overview

Temperature is a time-dependent variable where previous observations can influence future values. This project uses an **LSTM network**, a type of Recurrent Neural Network (RNN), to learn temporal patterns from historical temperature data and forecast upcoming values.

The project demonstrates an end-to-end **Time Series Forecasting + Deep Learning** workflow.

## ✨ Features

* 📊 Historical temperature visualization
* 🔄 Time-series data preprocessing
* 📐 Min-Max normalization
* 🧩 Sequential window/sequence generation
* 🧠 LSTM-based deep learning model
* 📈 Actual vs. predicted temperature visualization
* 🔮 Future temperature prediction
* 🌙 Dark-themed Streamlit interface
* 📉 Interactive charts and prediction results
* ⚡ Simple and user-friendly web interface

## 🛠️ Tech Stack

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| Python             | Core programming language   |
| TensorFlow / Keras | LSTM model development      |
| NumPy              | Numerical computation       |
| Pandas             | Data processing             |
| Scikit-learn       | Data normalization          |
| Matplotlib         | Visualization               |
| Streamlit          | Interactive web application |

## 📁 Project Structure

```text
LSTM-Temperature-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   └── temperature_lstm.keras
│
└── data/
    └── temperature.csv
```

## 🧠 How It Works

```text
Historical Temperature Data
          ↓
     Data Cleaning
          ↓
    Min-Max Scaling
          ↓
   Create Time Windows
          ↓
     LSTM Network
          ↓
      Model Training
          ↓
    Temperature Forecast
          ↓
 Streamlit Visualization
```

The historical temperature values are normalized using `MinMaxScaler`. The data is then converted into sequences, where previous temperature observations are used as input to predict the next temperature value.

The trained LSTM model learns temporal dependencies and generates temperature predictions from new sequences.

## 📊 Model Architecture

The model uses an LSTM-based architecture designed for sequential time-series data.

```text
Input Sequence
      ↓
LSTM Layer
      ↓
Dense Layer
      ↓
Temperature Prediction
```

LSTM is particularly useful for this problem because it can retain information from previous time steps and learn patterns across sequential observations.

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/LSTM-Temperature-Prediction.git
cd LSTM-Temperature-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
streamlit
```

## 📈 Results

The application provides visual comparisons between historical/actual temperature values and the values predicted by the LSTM model. It also allows users to explore temperature trends and generate forecasts through an interactive interface.

## 🎯 Learning Outcomes

Through this project, I practiced:

* Time-series data preprocessing
* Feature scaling and normalization
* Sequence/window creation
* Recurrent Neural Networks
* LSTM architecture
* Deep learning model training
* Model evaluation
* Data visualization
* Streamlit application development
* Deploying a machine learning model as an interactive application

## 🔮 Future Improvements

* Add multi-step forecasting
* Experiment with GRU and Bidirectional LSTM
* Add weather-related features such as humidity and pressure
* Compare LSTM with XGBoost and traditional forecasting models
* Add prediction confidence intervals
* Deploy the application online
* Improve forecasting accuracy with hyperparameter tuning

## 👩‍💻 Author

**Roshni Chauhan**

BCA Student | Data Science & Machine Learning Enthusiast

### ⭐ If you find this project useful

Give the repository a ⭐ and feel free to explore, improve, or contribute to the project.


