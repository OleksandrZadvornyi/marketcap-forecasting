import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")  # Suppress ARIMA convergence warnings

# Load the data
data = pd.read_csv('../market_cap_data/Apple_market_cap.csv', parse_dates=['Date'], index_col='Date')

# Extract the market cap series
market_cap = data['Market Cap (Billion USD)']

# Check for NaN values in the data
if market_cap.isna().any():
    print("Warning: NaN values found in market cap data. Dropping NaNs...")
    market_cap = market_cap.dropna()

# Function to test stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] < 0.05:
        print("Series is stationary (p-value < 0.05)")
    else:
        print("Series is non-stationary (p-value >= 0.05)")

# Test stationarity of the original and differenced series
print("Stationarity Test for Original Series:")
test_stationarity(market_cap)
print("\nStationarity Test for First Differenced Series:")
test_stationarity(market_cap.diff().dropna())
print("\nStationarity Test for Second Differenced Series:")
test_stationarity(market_cap.diff().diff().dropna())

# Split into train and test sets (last 12 months for validation)
train_size = int(len(market_cap) - 12)
train, test = market_cap[:train_size], market_cap[train_size:]

# Check lengths and indices
print(f"\nTrain size: {len(train)}, Test size: {len(test)}")
print("Test indices:", test.index)
print("Test values:", test.values)

# Fit ARIMA(1,2,3) model
print("\nFitting ARIMA(1,2,3) model...")
try:
    model = ARIMA(train, order=(1,2,3))
    model_fit = model.fit()
except Exception as e:
    print(f"Error fitting ARIMA model: {e}")
    exit()

# Forecast for the test period
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(start=test.index[0], periods=12, freq='MS')

# Convert to NumPy arrays to avoid index issues
test_np = test.values
forecast_np = forecast.values

# Check for NaN in forecast or test
if np.any(np.isnan(test_np)) or np.any(np.isnan(forecast_np)):
    print("Error: NaN values in test or forecast!")
    print(f"Test: {test_np}")
    print(f"Forecast: {forecast_np}")
    exit()

# Ensure same length
if len(test_np) != len(forecast_np):
    print(f"Error: Length mismatch! Test length: {len(test_np)}, Forecast length: {len(forecast_np)}")
    exit()

# Calculate evaluation metrics
# RMSE
rmse = sqrt(mean_squared_error(test_np, forecast_np))

# MAE
mae = np.mean(np.abs(test_np - forecast_np))

# MASE
naive_forecast = test.shift(1).iloc[1:].values  # Naive forecast: previous value
test_aligned = test_np[1:]  # Align with naive forecast
forecast_aligned = forecast_np[1:]  # Align with naive forecast
if len(test_aligned) == len(naive_forecast) and len(forecast_aligned) == len(naive_forecast):
    naive_errors = np.abs(test_aligned - naive_forecast)
    scaling_factor = np.mean(naive_errors) if len(naive_errors) > 0 and not np.any(np.isnan(naive_errors)) else np.nan
    mase = mae / scaling_factor if scaling_factor and not np.isnan(scaling_factor) else np.nan
else:
    print("Error: Length mismatch in MASE calculation!")
    mase = np.nan

# sMAPE
smape = np.mean(2 * np.abs(test_np - forecast_np) / (np.abs(test_np) + np.abs(forecast_np))) * 100 if np.all(np.abs(test_np) + np.abs(forecast_np) > 0) else np.nan

# MAPE
mape = np.mean(np.abs((test_np - forecast_np) / test_np)) * 100 if np.all(test_np != 0) else np.nan

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"RMSE: {rmse:.3f} Billion USD")
print(f"MAE: {mae:.3f} Billion USD")
print(f"MASE: {mase:.3f}")
print(f"sMAPE: {smape:.3f}%")
print(f"MAPE: {mape:.3f}%")

# Debug: Print test, forecast, and naive errors
print("\nDebug Info:")
print("Test values:", test_np)
print("Forecast values:", forecast_np)
print("Naive forecast values:", naive_forecast)
print("Naive errors:", naive_errors)
print("Scaling factor (mean naive error):", scaling_factor)

# Print model summary
print("\nModel Summary:")
print(model_fit.summary())

# Plot actual vs forecast for test period
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Market Cap')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('Apple Market Cap Forecast vs Actuals (ARIMA(1,2,3))')
plt.ylabel('Market Cap (Billion USD)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Residual diagnostics
residuals = model_fit.resid
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title('Residuals of ARIMA(1,2,3) Model')
plt.grid(True)
plt.subplot(212)
plt.acorr(residuals, maxlags=20)
plt.title('Autocorrelation of Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

# Forecast the next 12 months (future)
future_forecast = model_fit.forecast(steps=12)
future_index = pd.date_range(start=market_cap.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Plot historical data + test period + future forecast
plt.figure(figsize=(12, 6))
# Plot training data
plt.plot(train.index, train, label='Training Data', color='blue')
# Plot actual test data (May 2024–April 2025)
plt.plot(test.index, test, label='Actual Test Data', color='orange')
# Plot test period forecast (May 2024–April 2025)
plt.plot(forecast_index, forecast, label='Test Period Forecast', color='red', linestyle='--')
# Plot future forecast (May 2025–April 2026)
plt.plot(future_index, future_forecast, label='Future Forecast', color='green')
plt.title('Apple Market Cap - Historical and 12 Month Forecast (ARIMA(1,2,3))')
plt.ylabel('Market Cap (Billion USD)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Print the forecasted values
forecast_df = pd.DataFrame({
    'Date': future_index,
    'Forecasted Market Cap (Billion USD)': future_forecast
})
print("\n12-Month Forecast:")
print(forecast_df.round(2))