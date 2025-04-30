import yfinance as yf
from datetime import datetime
import pandas as pd
import os

# 1. Load companies from CSV
csv_path = "Companies_ranked_by_Market_Cap.csv"  # Adjust if needed
df = pd.read_csv(csv_path)

# 2. Keep first max_files rows and drop any with missing tickers
max_files = 1000
top = df[['Symbol', 'Name']].dropna().head(max_files)

# 3. Create dictionary: {symbol: company_name}
companies = dict(zip(top['Symbol'], top['Name']))

# 4. Set up output directory and date range
output_dir = "market_cap_data"
os.makedirs(output_dir, exist_ok=True)

# Define the start and end dates for historical data
start_date = "2000-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# 5. Fetch and save data
for ticker, name in companies.items():
    try:
        print(f"Fetching {name} ({ticker})...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval="1mo")

        if hist.empty:
            print(f"  No data found for {ticker}")
            continue

        shares_outstanding = stock.info.get("sharesOutstanding", None)
        if shares_outstanding is None:
            print(f"  Missing shares outstanding for {ticker}")
            continue

        hist = hist[["Close"]].rename(columns={"Close": "Stock Price"})
        hist["Market Cap (Billion USD)"] = hist["Stock Price"] * shares_outstanding / 1e9
        hist["Company"] = name

        filename = f"{name.replace(' ', '_').replace('/', '_')}_market_cap.csv"
        hist.to_csv(os.path.join(output_dir, filename))
        print(f"  Saved to {filename}")
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")