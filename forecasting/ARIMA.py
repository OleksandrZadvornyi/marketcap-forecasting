import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('market_cap_data/Apple_market_cap.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filter for a specific company (3M in this example)
company_data = data[data['Company'] == 'Apple'][['Market Cap (Billion USD)']]

# Plot the time series
plt.figure(figsize=(12,6))
plt.plot(company_data['Market Cap (Billion USD)'])
plt.title('Apple Market Capitalization (Billion USD)')
plt.xlabel('Date')
plt.ylabel('Market Cap (Billion USD)')
plt.grid(True)
plt.show()

# Stationarity check
adf_test = adfuller(company_data['Market Cap (Billion USD)'])
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

# ACF and PACF plots
plot_acf(company_data['Market Cap (Billion USD)'], lags=20)
plt.show()
plot_pacf(company_data['Market Cap (Billion USD)'], lags=20)
plt.show()

# Split the data into train and test
train_size = int(len(company_data) * 0.8)
train, test = company_data[0:train_size], company_data[train_size:len(company_data)]

# Fit the ARIMA model (parameters may need adjustment)
model = ARIMA(train['Market Cap (Billion USD)'], order=(1,1,1))
model_fit = model.fit()

# Forecast on the test dataset
test_forecast = model_fit.get_forecast(steps=len(test))
test_forecast_series = pd.Series(test_forecast.predicted_mean, index=test.index)

# Calculate RMSE
mse = mean_squared_error(test['Market Cap (Billion USD)'], test_forecast_series)
rmse = mse ** 0.5

# Create comparison plot
plt.figure(figsize=(14,7))
plt.plot(train['Market Cap (Billion USD)'], label='Training Data')
plt.plot(test['Market Cap (Billion USD)'], label='Actual Test Data', color='blue')
plt.plot(test_forecast_series, label='Forecasted Data', color='red')
plt.fill_between(test.index, 
                test_forecast.conf_int().iloc[:, 0], 
                test_forecast.conf_int().iloc[:, 1], 
                color='k', alpha=.15)
plt.title('3M Market Cap: ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Market Cap (Billion USD)')
plt.legend()
plt.grid(True)
plt.show()

print(f'Root Mean Squared Error: {rmse:.2f} Billion USD')