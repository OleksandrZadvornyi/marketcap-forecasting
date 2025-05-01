import pandas as pd
import glob
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from gluonts.itertools import Map
import os
import shutil

# Load all CSVs
csv_files = glob.glob("market_cap_data/*.csv")
print(f"Found {len(csv_files)} market cap files")

# Load and clean each file
company_dfs = []
for file in csv_files:
    df = pd.read_csv(file, parse_dates=["Date"])
    df = df.rename(columns={"Market Cap (Billion USD)": "MARKET_CAP"})
    company = os.path.basename(file).replace("_market_cap.csv", "")
    df["Company"] = company
    df = df.sort_values("Date").dropna(subset=["MARKET_CAP"])
    company_dfs.append(df)

# Combine all into one DataFrame
combined_df = pd.concat(company_dfs)
combined_df = combined_df.reset_index(drop=True)

# Split per company
company_groups = {k: v for k, v in combined_df.groupby("Company")}

# Time-based split
prediction_length = 12
def split_time_series(data, prediction_length=prediction_length):
    train, val, test = {}, {}, {}
    for company, df in data.items():
        df = df.sort_values("Date")
        if len(df) < 2 * prediction_length:
            continue
        n = len(df)
        train_end = n - 2 * prediction_length
        val_end = n - prediction_length
        train[company] = df.iloc[:train_end]
        val[company] = df.iloc[:val_end]
        test[company] = df
    return train, val, test

train_data, val_data, test_data = split_time_series(company_groups)

# Convert to GluonTS format
def to_gluonts_format(data_dict):
    output = []
    id_map = {k: i for i, k in enumerate(data_dict.keys())}
    for k, df in data_dict.items():
        #print(k)
        start = df["Date"].iloc[0]
        target = df["MARKET_CAP"].values.astype("float32")
        ts = {
            "start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "target": target,
            "feat_static_cat": [id_map[k]],
            "item_id": k
        }
        output.append(ts)
    return output

train_gl = to_gluonts_format(train_data)
val_gl = to_gluonts_format(val_data)
test_gl = to_gluonts_format(test_data)

# Process timestamps for HuggingFace
class ProcessStartField:
    ts_id = 0
    def __call__(self, data):
        data["start"] = pd.Timestamp(data["start"]).to_pydatetime()
        data["feat_static_cat"] = [self.ts_id]
        self.ts_id += 1
        return data

process = ProcessStartField()
train_ds = list(Map(process, train_gl))
process = ProcessStartField()
val_ds = list(Map(process, val_gl))
process = ProcessStartField()
test_ds = list(Map(process, test_gl))

# HuggingFace features
features = Features({
    "start": Value("timestamp[s]"),
    "target": Sequence(Value("float32")),
    "feat_static_cat": Sequence(Value("int64")),
    "item_id": Value("string")
})

train_dataset = Dataset.from_list(train_ds, features=features)
val_dataset = Dataset.from_list(val_ds, features=features)
test_dataset = Dataset.from_list(test_ds, features=features)

# Save datasets
output_dir = "prepared_marketcap_dataset"

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    
os.makedirs(output_dir)
dataset_dict.save_to_disk(f"{output_dir}/dataset")

with open(f"{output_dir}/metadata.txt", "w") as f:
    f.write("freq=M\n")  # Monthly
    f.write(f"prediction_length={prediction_length}\n")

print("Done! Saved to:", output_dir)
