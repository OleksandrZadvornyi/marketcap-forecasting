import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder containing the CSV files
data_folder = "market_cap_data"
plots_folder = "market_cap_data/plots"

# List of specific files to read
target_files = {
    "Apple_market_cap.csv",
    "Microsoft_market_cap.csv",
    "NVIDIA_market_cap.csv",
    "Tencent_market_cap.csv",
    "ICBC_market_cap.csv",
    "Alibaba_market_cap.csv"
}

# Collect only the target CSV files from the folder
csv_files = [f for f in os.listdir(data_folder) if f in target_files]

# Prepare a color palette for consistency
colors = plt.cm.get_cmap("tab10", len(csv_files))

# For combined plot
plt.figure(figsize=(12, 8))

for idx, filename in enumerate(csv_files):
    filepath = os.path.join(data_folder, filename)
    df = pd.read_csv(filepath, parse_dates=["Date"])

    # Plot individual line chart
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Market Cap (Billion USD)"], label=filename.replace("_market_cap.csv", ""), color=colors(idx))
    plt.title(f"Market Capitalization of {filename.replace('_market_cap.csv', '')}")
    plt.xlabel("Date")
    plt.ylabel("Market Cap (Billion USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plots_folder, filename.replace(".csv", "_plot.png")))
    plt.close()

    # Add to combined plot
    plt.plot(df["Date"], df["Market Cap (Billion USD)"], label=filename.replace("_market_cap.csv", ""), color=colors(idx))

# Show combined chart
plt.title("Market Capitalization Comparison")
plt.xlabel("Date")
plt.ylabel("Market Cap (Billion USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "combined_market_cap_plot.png"))
plt.show()
