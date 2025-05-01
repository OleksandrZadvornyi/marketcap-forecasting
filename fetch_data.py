import yfinance as yf
from datetime import datetime
import pandas as pd
import os
import time
import numpy as np

# 1. Load companies from CSV
csv_path = "Companies_ranked_by_Market_Cap.csv"  # Adjust if needed
df = pd.read_csv(csv_path)

# 2. Target number of successful CSV files
target_files = 1000

# 3. Create dictionary of all available companies: {symbol: company_name}
companies = dict(zip(df['Symbol'].dropna(), df['Name'].dropna()))

# 4. Set up output directory and date range
output_dir = "market_cap_data"
os.makedirs(output_dir, exist_ok=True)

# Define the start and end dates for historical data
start_date = "2000-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# 5. Keep track of successful files
successful_files = 0
processed_tickers = set()

# 6. Process companies until we have enough files
for ticker, name in companies.items():
    # Skip if we already have enough files
    if successful_files >= target_files:
        break
        
    # Skip if we've already processed this ticker
    if ticker in processed_tickers:
        continue
    
    processed_tickers.add(ticker)
    
    try:
        print(f"Fetching {name} ({ticker}) - Progress: {successful_files}/{target_files}")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval="1mo")

        if hist.empty:
            print(f"  No data found for {ticker}")
            continue

        # Check for minimum 5 years of data (60 months)
        if len(hist) < 60:
            print(f"  Insufficient data for {ticker} - only {len(hist)} months available")
            continue

        # Get company info for static features
        info = stock.info
        shares_outstanding = info.get("sharesOutstanding", None)
        if shares_outstanding is None:
            print(f"  Missing shares outstanding for {ticker}")
            continue

        # Basic price data
        hist = hist[["Close"]].rename(columns={"Close": "Stock Price"})
        hist["Market Cap (Billion USD)"] = hist["Stock Price"] * shares_outstanding / 1e9
        
        # Company name as a static feature
        hist["Company"] = name
        
        # Add sector and industry as categorical static features
        hist["Sector"] = info.get("sector", "Unknown")
        hist["Industry"] = info.get("industry", "Unknown")
        
        # Add financial metrics as static features
        # Using the first available value or default for missing ones
        hist["PE_Ratio"] = info.get("trailingPE", np.nan)
        hist["PB_Ratio"] = info.get("priceToBook", np.nan)
        hist["Profit_Margin"] = info.get("profitMargins", np.nan)
        hist["ROE"] = info.get("returnOnEquity", np.nan)
        hist["ROA"] = info.get("returnOnAssets", np.nan)
        hist["Debt_to_Equity"] = info.get("debtToEquity", np.nan)
        
        # Add company age (years since IPO) if available
        if "firstTradeDateEpochUtc" in info and info["firstTradeDateEpochUtc"] is not None:
            ipo_date = datetime.fromtimestamp(info["firstTradeDateEpochUtc"])
            hist["Company_Age_Years"] = (datetime.now() - ipo_date).days / 365.25
        else:
            hist["Company_Age_Years"] = np.nan
            
        # Market cap category (Small, Mid, Large, Mega)
        market_cap = shares_outstanding * hist["Stock Price"].iloc[-1] if not hist.empty else 0
        if market_cap < 2e9:  # Less than $2B
            cap_category = "Small Cap"
        elif market_cap < 10e9:  # $2B to $10B
            cap_category = "Mid Cap"
        elif market_cap < 200e9:  # $10B to $200B
            cap_category = "Large Cap"
        else:  # $200B+
            cap_category = "Mega Cap"
        hist["Market_Cap_Category"] = cap_category

        filename = f"{name.replace(' ', '_').replace('/', '_')}_market_cap.csv"
        file_path = os.path.join(output_dir, filename)
        hist.to_csv(file_path)
        
        successful_files += 1
        print(f"  Saved to {filename} ({successful_files}/{target_files})")
        
        # Add a small delay to avoid getting rate-limited
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")

print(f"\nCompleted! Generated {successful_files} market cap CSV files.")