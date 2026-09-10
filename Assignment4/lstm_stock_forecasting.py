import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Configuration
# -----------------------------
TICKER = "AAPL"
PERIOD = "5y"
LOOKBACK = 60
TRAIN_RATIO = 0.80
EPOCHS = 20
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 1. Load dataset
# -----------------------------
csv_file = "stock_data.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if "Close" not in df.columns:
        raise ValueError("stock_data.csv must contain a 'Close' column.")
    close_prices = df["Close"].astype(float).dropna().values.reshape(-1, 1)
    dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else np.arange(len(close_prices))
else:
    try:
        import yfinance as yf
        df = yf.download(TICKER, period=PERIOD, progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError("No data downloaded.")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close_prices = close.astype(float).dropna().values.reshape(-1, 1)
        dates = df.index[-len(close_prices):]
        df = pd.DataFrame({"Date": dates, "Close": close_prices.ravel()})
        df.to_csv(csv_file, index=False)
        print(f"Downloaded and saved dataset to {csv_file}")
    except Exception as e:
        raise RuntimeError(
            "Could not download data. Install yfinance and/or provide stock_data.csv "
            "with a Close column."
        ) from e

print(f"Number of observations: {len(close_prices)}")

if len(close_prices) <= LOOKBACK + 20:
    raise ValueError("Dataset is too small. Provide more observations.")

# -----------------------------
# 2. Train/test split
# -----------------------------
train_size = int(len(close_prices) * TRAIN_RATIO)

train_data = close_prices[:train_size]
test_data = close_prices[train_size - LOOKBACK:]

# -----------------------------
# 3. Normalize
# Fit scaler ONLY on training data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# -----------------------------
# 4. Create sequences
# -----------------------------
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_test, y_test = create_sequences(test_scaled, LOOKBACK)

# LSTM expects: samples, timesteps, features
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# -----------------------------
# 5. Build LSTM model
# -----------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),

    LSTM(50),
    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# -----------------------------
# 6. Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    shuffle=False,
    verbose=1
)

# -----------------------------
# 7. Prediction
# -----------------------------
pred_scaled = model.predict(X_test, verbose=0)

predicted = scaler.inverse_transform(pred_scaled).ravel()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

# -----------------------------
# 8. Evaluation
# -----------------------------
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Avoid division by zero for MAPE
nonzero = actual != 0
mape = np.mean(
    np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])
) * 100

print("\n===== Evaluation Metrics =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

# -----------------------------
# 9. Plot training loss
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()

# -----------------------------
# 10. Plot actual vs predicted
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual Price")
plt.plot(predicted, label="Predicted Price")
plt.title(f"{TICKER} Stock Price: Actual vs Predicted")
plt.xlabel("Test Time Step")
plt.ylabel("Closing Price")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# -----------------------------
# 11. Save model
# -----------------------------
model.save("lstm_stock_model.keras")
print("\nSaved:")
print("- actual_vs_predicted.png")
print("- training_loss.png")
print("- lstm_stock_model.keras")
