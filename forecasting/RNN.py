import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Function to calculate MASE with handling for zero denominator
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1) if n > 1 else 0
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d if d > 1e-10 else np.nan

# Function to calculate sMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    errors = np.abs(y_true - y_pred) / denominator
    return 2.0 * errors.mean() * 100  # Convert to percentage

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    errors = np.abs((y_true - y_pred) / y_true)
    return errors.mean() * 100  # Convert to percentage

# Load and preprocess data
data = pd.read_csv('../market_cap_data/Apple_market_cap.csv', parse_dates=['Date'], index_col='Date')
data = data.sort_index()

# Apply log transformation to market cap
market_cap = np.log1p(data['Market Cap (Billion USD)']).values.reshape(-1, 1)  # log1p for stability with zeros

# Normalize the data
scaler = MinMaxScaler()
market_cap_scaled = scaler.fit_transform(market_cap)

# Create sequences for RNN
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # Use 12 months as sequence length
X, y = create_sequences(market_cap_scaled, seq_length)

# Split into train and test sets (80-20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build RNN model with dropout
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# Evaluate on test set
y_pred_scaled = model.predict(X_test, verbose=0)
y_test_orig = np.expm1(scaler.inverse_transform(y_test))  # Inverse log transformation
y_pred_orig = np.expm1(scaler.inverse_transform(y_pred_scaled))  # Inverse log transformation

# Calculate metrics
mase = mean_absolute_scaled_error(y_test_orig, y_pred_orig, np.expm1(scaler.inverse_transform(y_train)))
smape = symmetric_mean_absolute_percentage_error(y_test_orig, y_pred_orig)
mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)
print(f"Test Set Metrics:")
print(f"MASE: {mase:.4f}" if not np.isnan(mase) else "MASE: NaN (due to minimal variation in training data)")
print(f"sMAPE: {smape:.4f}%")
print(f"MAPE: {mape:.4f}%")

# Forecast next 12 months
last_sequence = market_cap_scaled[-seq_length:].reshape(1, seq_length, 1)
forecast = []
last_value = np.expm1(scaler.inverse_transform([market_cap_scaled[-1]])[0, 0])  # Last historical value
max_growth_rate = 1.10  # Maximum 10% growth per month
for i in range(12):
    pred = model.predict(last_sequence, verbose=0)
    forecast.append(pred[0, 0])
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = pred[0, 0]
    # Cap the forecast to prevent extreme values
    pred_value = np.expm1(scaler.inverse_transform(pred)[0, 0])
    if pred_value > last_value * (max_growth_rate ** (i + 1)):
        pred_value = last_value * (max_growth_rate ** (i + 1))
        pred_scaled = scaler.transform(np.log1p([[pred_value]]) )[0, 0]
        last_sequence[0, -1, 0] = pred_scaled

# Inverse transform forecast
forecast = np.expm1(scaler.inverse_transform(np.array(forecast).reshape(-1, 1)))

# Generate future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=12, freq='MS')

# Plot historical data (last 5 years), test predictions, and forecast
plt.figure(figsize=(12, 6))
# Filter historical data to last 5 years
start_date = data.index[-1] - pd.Timedelta(days=5*365)
recent_data = data[start_date:]
plt.plot(recent_data.index, recent_data['Market Cap (Billion USD)'], label='Historical Market Cap (Last 5 Years)', color='blue')
# Plot test predictions
test_dates = data.index[seq_length + train_size:seq_length + len(data)]
plt.plot(test_dates, y_pred_orig, label='Test Predictions', color='green', linestyle='--')
# Plot forecast
plt.plot(future_dates, forecast, label='Forecast (May 2025 - Apr 2026)', color='red', linestyle='-.')
plt.title("Apple Market Cap Forecast (RNN)")
plt.xlabel("Date")
plt.ylabel("Market Cap (Billion USD)")
# Set y-axis limit based on historical data
max_y = data['Market Cap (Billion USD)'].max() * 1.5  # 50% above max historical value
plt.ylim(0, max_y)
plt.legend()
plt.grid(True)
plt.savefig('forecast_plot_rnn.png')
print("Plot saved to forecast_plot_rnn.png")