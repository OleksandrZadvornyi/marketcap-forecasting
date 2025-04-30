import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import datetime
warnings.filterwarnings("ignore")

# Create output directory
output_folder = os.path.join("forecasts", "arima_sarima_forecasts")
os.makedirs(output_folder, exist_ok=True)

# Parameters
forecast_steps = 12
selected_companies = ["Apple", "Microsoft", "NVIDIA", "Tencent", "ICBC", "Alibaba"]
data_folder = "market_cap_data"

# Load and clean all CSVs
def load_marketcap_data(folder):
    csv_files = glob.glob(f"{folder}/*.csv")
    print(f"Found {len(csv_files)} market cap files")
    company_dfs = []
    
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=["Date"])
        df = df.rename(columns={"Market Cap (Billion USD)": "MARKET_CAP"})
        company = os.path.basename(file).replace("_market_cap.csv", "")
        df["Company"] = company
        df = df.sort_values("Date").dropna(subset=["MARKET_CAP"])
        company_dfs.append(df)
    
    combined_df = pd.concat(company_dfs).reset_index(drop=True)
    return {k: v for k, v in combined_df.groupby("Company")}

# Function to optimize ARIMA parameters using grid search
def optimize_arima(train_data):
    best_score, best_params = float("inf"), (1, 1, 1)  # Default to (1,1,1) if nothing better is found
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train_data, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_score:
                best_score = aic
                best_params = (p, d, q)
        except Exception:
            continue
    
    print(f"Best ARIMA parameters: {best_params} with AIC: {best_score}")
    return best_params

# Function to optimize SARIMA parameters
def optimize_sarima(train_data, seasonal_period=12):
    best_score, best_params = float("inf"), (1, 1, 1, 1, 1, 1)  # Default if nothing better is found
    p_values = d_values = q_values = range(0, 2)
    P_values = D_values = Q_values = range(0, 2)
    
    # Limited parameter search to prevent excessive runtime
    param_combinations = list(itertools.product(
        p_values, d_values, q_values,
        P_values, D_values, Q_values
    ))[:10]  # Limit combinations
    
    for params in param_combinations:
        p, d, q, P, D, Q = params
        try:
            model = SARIMAX(
                train_data, 
                order=(p, d, q), 
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            aic = model_fit.aic
            if aic < best_score:
                best_score = aic
                best_params = (p, d, q, P, D, Q)
        except Exception:
            continue
    
    print(f"Best SARIMA parameters: {best_params} with AIC: {best_score}")
    return best_params

# Split data for validation
def train_test_split(data, test_size=12):
    if len(data) <= test_size:
        test_size = max(1, int(len(data) * 0.2))  # Use 20% for testing if not enough data
    return data[:-test_size], data[-test_size:]

# Forecast using ARIMA model
def forecast_with_arima(company, df):
    df = df.sort_values("Date").set_index("Date")
    ts = df["MARKET_CAP"]
    
    # Create train/test sets
    train_data, test_data = train_test_split(ts)
    
    # Optimize parameters
    p, d, q = optimize_arima(train_data)
    
    # Fit model on training data
    try:
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast on test data
        forecast = model_fit.forecast(steps=len(test_data))
        
        # Calculate validation error
        rmse = sqrt(mean_squared_error(test_data, forecast))
        
        # Fit full model and forecast future
        full_model = ARIMA(ts, order=(p, d, q))
        full_model_fit = full_model.fit()
        future_forecast = full_model_fit.forecast(steps=forecast_steps)
        
        # Generate future date index
        future_index = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_steps, 
            freq='MS'
        )
        future_forecast.index = future_index
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(ts.index, ts, label="Historical Data", color="blue")
        
        # Plot validation forecast
        plt.plot(test_data.index, forecast, label="Validation Forecast", color="green", linestyle='--')
        plt.plot(test_data.index, test_data, label="Validation Data", color="black", marker='.', linestyle='')
        
        # Plot future forecast
        plt.plot(future_forecast.index, future_forecast, label="Future Forecast", color="red")
        
        plt.title(f"ARIMA({p},{d},{q}) Forecast - {company}\nValidation RMSE: {rmse:.2f}")
        plt.xlabel("Date")
        plt.ylabel("Market Cap (Billion USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(output_folder, f"{company}_ARIMA.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved ARIMA plot to {plot_filename}")
        
        return rmse, f"ARIMA({p},{d},{q})"
    
    except Exception as e:
        print(f"Error in ARIMA for {company}: {e}")
        return float('inf'), "ARIMA failed"

# Forecast using SARIMA model
def forecast_with_sarima(company, df, seasonal_period=12):
    df = df.sort_values("Date").set_index("Date")
    ts = df["MARKET_CAP"]
    
    # Create train/test sets
    train_data, test_data = train_test_split(ts)
    
    # Optimize parameters
    try:
        p, d, q, P, D, Q = optimize_sarima(train_data, seasonal_period)
        
        # Fit model on training data
        model = SARIMAX(
            train_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        
        # Forecast on test data
        forecast = model_fit.forecast(steps=len(test_data))
        
        # Calculate validation error
        rmse = sqrt(mean_squared_error(test_data, forecast))
        
        # Fit full model and forecast future
        full_model = SARIMAX(
            ts,
            order=(p, d, q),
            seasonal_order=(P, D, Q, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        full_model_fit = full_model.fit(disp=False)
        future_forecast = full_model_fit.forecast(steps=forecast_steps)
        
        # Generate future date index
        future_index = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_steps, 
            freq='MS'
        )
        future_forecast.index = future_index
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(ts.index, ts, label="Historical Data", color="blue")
        
        # Plot validation forecast
        plt.plot(test_data.index, forecast, label="Validation Forecast", color="green", linestyle='--')
        plt.plot(test_data.index, test_data, label="Validation Data", color="black", marker='.', linestyle='')
        
        # Plot future forecast
        plt.plot(future_forecast.index, future_forecast, label="Future Forecast", color="red")
        
        # Add confidence intervals for future forecast
        forecast_obj = full_model_fit.get_forecast(steps=forecast_steps)
        conf_int = forecast_obj.conf_int()
        plt.fill_between(
            future_index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='red', 
            alpha=0.2, 
            label="95% Confidence Interval"
        )
        
        plt.title(f"SARIMA({p},{d},{q})({P},{D},{Q},{seasonal_period}) - {company}\nValidation RMSE: {rmse:.2f}")
        plt.xlabel("Date")
        plt.ylabel("Market Cap (Billion USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(output_folder, f"{company}_SARIMA.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved SARIMA plot to {plot_filename}")
        
        return rmse, f"SARIMA({p},{d},{q})({P},{D},{Q},{seasonal_period})"
    
    except Exception as e:
        print(f"Error in SARIMA for {company}: {e}")
        return float('inf'), "SARIMA failed"

# Run analysis
company_data = load_marketcap_data(data_folder)
results = []

for company in selected_companies:
    if company in company_data:
        print(f"\n=== Analyzing {company} ===")
        
        # Try both SARIMA and ARIMA
        sarima_rmse, sarima_model = forecast_with_sarima(company, company_data[company])
        arima_rmse, arima_model = forecast_with_arima(company, company_data[company])
        
        # Record results
        if sarima_rmse != float('inf') or arima_rmse != float('inf'):
            results.append({
                'Company': company,
                'SARIMA_RMSE': sarima_rmse if sarima_rmse != float('inf') else "Failed",
                'ARIMA_RMSE': arima_rmse if arima_rmse != float('inf') else "Failed",
                'Best_Model': 'SARIMA' if sarima_rmse < arima_rmse else 'ARIMA',
                'Best_RMSE': min(sarima_rmse, arima_rmse) if min(sarima_rmse, arima_rmse) != float('inf') else "All Failed"
            })
    else:
        print(f"Company '{company}' not found in dataset.")

# Print and save summary results
if results:
    print("\n=== Model Comparison Summary ===")
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Save results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(output_folder, f"forecast_results_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")
    
    # Save as formatted markdown table
    results_md = os.path.join(output_folder, f"forecast_results_{timestamp}.md")
    with open(results_md, 'w') as f:
        f.write("# Time Series Forecasting Results\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Model Details\n\n")
        f.write("- ARIMA: Autoregressive Integrated Moving Average\n")
        f.write("- SARIMA: Seasonal ARIMA\n")
        f.write("- RMSE: Root Mean Square Error (lower is better)\n")
    print(f"Formatted results saved to {results_md}")
    
    # Save summary HTML report with embedded images
    html_report = os.path.join(output_folder, f"forecast_report_{timestamp}.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Time Series Forecasting Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .image-gallery {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .image-container {{ margin-bottom: 30px; }}
            h1, h2, h3 {{ color: #333; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .best {{ font-weight: bold; color: green; }}
        </style>
    </head>
    <body>
        <h1>Time Series Forecasting Report</h1>
        <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Results Summary</h2>
        <table>
            <tr>
                <th>Company</th>
                <th>SARIMA RMSE</th>
                <th>ARIMA RMSE</th>
                <th>Best Model</th>
                <th>Best RMSE</th>
            </tr>
    """
    
    # Add table rows
    for _, row in results_df.iterrows():
        sarima_class = "best" if row['Best_Model'] == 'SARIMA' else ""
        arima_class = "best" if row['Best_Model'] == 'ARIMA' else ""
        
        html_content += f"""
            <tr>
                <td>{row['Company']}</td>
                <td class="{sarima_class}">{row['SARIMA_RMSE']}</td>
                <td class="{arima_class}">{row['ARIMA_RMSE']}</td>
                <td>{row['Best_Model']}</td>
                <td>{row['Best_RMSE']}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Forecast Visualizations</h2>
    """
    
    # Add images for each company
    for company in selected_companies:
        if company in company_data:
            html_content += f"""
            <div class="image-container">
                <h3>{company}</h3>
                <div class="image-gallery">
            """
            
            # Add ARIMA image if exists
            arima_img = f"{company}_ARIMA.png"
            arima_path = os.path.join(output_folder, arima_img)
            if os.path.exists(arima_path):
                html_content += f"""
                    <div>
                        <h4>ARIMA Model</h4>
                        <img src="{arima_img}" alt="{company} ARIMA forecast" style="max-width: 600px;">
                    </div>
                """
            
            # Add SARIMA image if exists
            sarima_img = f"{company}_SARIMA.png"
            sarima_path = os.path.join(output_folder, sarima_img)
            if os.path.exists(sarima_path):
                html_content += f"""
                    <div>
                        <h4>SARIMA Model</h4>
                        <img src="{sarima_img}" alt="{company} SARIMA forecast" style="max-width: 600px;">
                    </div>
                """
            
            html_content += """
                </div>
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(html_report, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {html_report}")