import os

from src.data_preprocessing import prepare_data
from src.model import build_model


DATA_PATH = "data/temperature.csv"
MODEL_PATH = "models/temperature_lstm.keras"
LOOKBACK = 5


def main():

    # Load and prepare data
    print("Loading and preparing data...")

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        DATA_PATH,
        LOOKBACK
    )

    print("\nTraining data shape:")
    print(X_train.shape)

    print("\nTesting data shape:")
    print(X_test.shape)

    # Build LSTM model
    model = build_model(LOOKBACK)

    print("\nModel Summary:")
    model.summary()

    # Train model
    print("\nTraining started...")

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save model
    model.save(MODEL_PATH)

    print("\nModel saved successfully!")
    print("Model path:", MODEL_PATH)

    # Evaluate model
    loss = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("Test Loss:", loss)


if __name__ == "__main__":
    main()