import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


# Define evaluation metrics
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def mean_absolute_scaled_error(y_true, y_pred, training_data):
    """Calculate Mean Absolute Scaled Error (MASE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = training_data.shape[0]
    d = np.abs(np.diff(training_data, axis=0)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


# List of companies to analyze
companies = ['Apple', 'NVIDIA', 'Microsoft', 'Tencent', 'ICBC', 'Alibaba']

# Function to create dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


# Function to train and evaluate model for each company
def train_and_evaluate(company_name):
    print(f"\nProcessing {company_name}...")
    
    # Read data from CSV
    csv_path = f'market_cap_data/{company_name}_market_cap.csv'
    df = pd.read_csv(csv_path)
    data = df['Market Cap (Billion USD)'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    
    predictions = model.predict(X_test)
    
    # Transform back to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_original = scaler.inverse_transform(predictions)
    training_data_original = scaler.inverse_transform(scaled_data[:train_size])
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(y_test_original, predictions_original)
    smape = symmetric_mean_absolute_percentage_error(y_test_original, predictions_original)
    mase = mean_absolute_scaled_error(y_test_original, predictions_original, training_data_original)
    
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape:.4f}%")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original, color='blue', label='Real Market Cap')
    plt.plot(predictions_original, color='red', label='Predicted Market Cap')
    plt.title(f'{company_name} Market Cap Prediction')
    plt.xlabel('Time')
    plt.ylabel('Market Cap (Billion USD)')
    plt.legend()
    plt.show()
    
    return mape, smape, mase


# Process each company
results = {}
for company in companies:
    results[company] = train_and_evaluate(company)

# Print summary of results
print("\nSummary of Results:")
print("-" * 80)
print(f"{'Company':<15} {'MAPE (%)':<15} {'sMAPE (%)':<15} {'MASE':<15}")
print("-" * 80)
for company, metrics in results.items():
    print(f"{company:<15} {metrics[0]:<15.4f} {metrics[1]:<15.4f} {metrics[2]:<15.4f}")