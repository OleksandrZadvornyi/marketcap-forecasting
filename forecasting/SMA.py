import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Calculating SMA
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# MASE calculation (using lag-1 naive forecast to avoid nan)
def calculate_mase(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    errors = np.abs(actual - forecast)
    naive_errors = np.abs(actual[1:] - actual[:-1])  # Lag-1 naive forecast
    mean_naive_error = np.mean(naive_errors)
    return np.mean(errors) / mean_naive_error if mean_naive_error != 0 else np.nan

# sMAPE calculation
def calculate_smape(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    errors = np.abs(actual - forecast)
    return np.mean(errors / denominator) * 100 if not np.any(denominator == 0) else np.nan

# MAPE calculation
def calculate_mape(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    errors = np.abs((actual - forecast) / actual)
    return np.mean(errors) * 100 if not np.any(actual == 0) else np.nan

# Loading data
df = pd.read_csv('../market_cap_data/Apple_market_cap.csv', parse_dates=['Date'])

# Ensuring data is sorted by date
df = df.sort_values('Date').dropna(subset=['Date', 'Market Cap (Billion USD)'])

# Converting market cap to numeric, handling invalid values
df['Market Cap (Billion USD)'] = pd.to_numeric(df['Market Cap (Billion USD)'], errors='coerce')
df = df.dropna(subset=['Market Cap (Billion USD)'])

# Setting date as index
df.set_index('Date', inplace=True)

# Calculating 12-month SMA for historical smoothing
window = 12
df['SMA'] = calculate_sma(df['Market Cap (Billion USD)'], window)

# Splitting data: last 12 months for validation, rest for training
validation_period = 12
train_df = df.iloc[:-validation_period]
validation_df = df.iloc[-validation_period:]

# Fitting linear trend to last 60 months (5 years) for forecasting
trend_period = 60  # Use last 5 years for trend
recent_df = df.iloc[-trend_period:]  # Last 60 months
x = np.arange(len(recent_df))  # Time indices (0, 1, ..., 59)
y = recent_df['Market Cap (Billion USD)'].values
# Fit linear model (degree 1)
slope, intercept = np.polyfit(x, y, 1)

# Forecasting next 12 months
last_date = df.index[-1]
forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, 13)]
# Extend time indices for forecast (60, 61, ..., 71)
forecast_x = np.arange(len(recent_df), len(recent_df) + 12)
forecast_values = intercept + slope * forecast_x  # Linear trend forecast
forecast_df = pd.DataFrame({'Market Cap (Billion USD)': forecast_values}, index=forecast_dates)

# Calculating metrics on validation set (using SMA for consistency with original)
actual = validation_df['Market Cap (Billion USD)'].values
forecast = [df['SMA'].iloc[-validation_period-1]] * len(actual)  # Using SMA from before validation period
mase = calculate_mase(actual, forecast)
smape = calculate_smape(actual, forecast)
mape = calculate_mape(actual, forecast)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Market Cap (Billion USD)'], label='Historical Market Cap', color='blue')
plt.plot(df.index, df['SMA'], label=f'{window}-Month SMA', color='orange')
plt.plot(forecast_df.index, forecast_df['Market Cap (Billion USD)'], label='Linear Trend Forecast', color='green', linestyle='--')
plt.title('Apple Market Capitalization with Linear Trend Forecast')
plt.xlabel('Date')
plt.ylabel('Market Cap (Billion USD)')
plt.legend()
plt.grid(True)
plt.savefig('apple_market_cap_forecast.png')

# Printing metrics
print(f"MASE: {mase:.2f}")
print(f"sMAPE: {smape:.2f}%")
print(f"MAPE: {mape:.2f}%")
print("Interesting Fact: Apple's market cap grew from $3.36 billion in Dec 2000 to over $3.5 trillion by Feb 2025, a remarkable ~1000x increase, showcasing its dominance in the tech sector.")