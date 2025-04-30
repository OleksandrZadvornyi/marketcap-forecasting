import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from datasets import load_from_disk
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    RemoveFields,
    TestSplitSampler,
    VstackFeatures,
    RenameFields,
    InstanceSplitter
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.dataset.loader import as_stacked_batches
from gluonts.time_feature import time_features_from_frequency_str
from transformers import PretrainedConfig
from typing import Optional

# Set paths and constants
model_dir = "marketcap_model_900_12"  # Update with your model directory
data_dir = "prepared_marketcap_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model configuration
config_path = os.path.join(model_dir, "config")
config = TimeSeriesTransformerConfig.from_pretrained(config_path)

# Load metadata
metadata = {}
with open(os.path.join(config_path, "metadata.txt"), "r") as f:
    for line in f:
        key, value = line.strip().split("=")
        metadata[key] = value

freq = metadata["freq"]
prediction_length = int(metadata["prediction_length"])

# Initialize model
model = TimeSeriesTransformerForPrediction(config)
model_path = os.path.join(model_dir, "time_series_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load dataset
dataset = load_from_disk(f"{data_dir}/dataset")
test_dataset = dataset["test"]

# Define target companies
target_companies = {"Apple", "Microsoft", "NVIDIA", "Tencent", "ICBC", "Alibaba"}

# Helper functions for data transformation
def create_transformation(freq: str, config: PretrainedConfig) -> Chain:
    """Create the transformation pipeline for time series data."""
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> InstanceSplitter:
    """Create an instance splitter based on the specified mode."""
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler,
        "validation": validation_sampler,
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    """Create a dataloader for generating forecasts from the end of time series."""
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, "test")

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def convert_to_gluonts_dataset(hf_dataset, freq):
    """Convert HuggingFace dataset format to GluonTS ListDataset format."""
    data = []
    for item in hf_dataset:
        data.append({
            FieldName.START: pd.Period(item["start"], freq=freq),
            FieldName.TARGET: item["target"],
            FieldName.FEAT_STATIC_CAT: [item["feat_static_cat"][0]],
            FieldName.ITEM_ID: item["item_id"]
        })
    return ListDataset(data, freq=freq)

def generate_forecasts(model, dataloader, device):
    """Generate forecasts using the model."""
    model.eval()
    forecasts = []
    item_ids = []
    forecast_start_dates = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            outputs = model.generate(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
            )
            
            # Store forecasts
            forecast_sequences = outputs.sequences.cpu().numpy()
            forecasts.append(forecast_sequences)
            
            # Extract metadata from batch if available
            if hasattr(batch, "item_id"):
                item_ids.extend(batch.item_id)
            if hasattr(batch, FieldName.FORECAST_START):
                forecast_start_dates.extend(batch[FieldName.FORECAST_START])
    
    if forecasts:
        forecasts = np.vstack(forecasts)
    
    return forecasts, item_ids, forecast_start_dates

def plot_forecast(ts_data, forecast, item_name, start_date, freq, prediction_length):
    """Plot the forecast along with historical data."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time index for historical data
    historical_end = pd.Period(start_date, freq=freq) + len(ts_data) - 1
    historical_index = pd.period_range(
        start=start_date,
        end=historical_end,
        freq=freq
    ).to_timestamp()
    
    # Create time index for forecast
    forecast_start = historical_end + 1
    forecast_index = pd.period_range(
        start=forecast_start,
        periods=prediction_length,
        freq=freq
    ).to_timestamp()
    
    # Plot historical data
    ax.plot(
        historical_index,
        ts_data,
        label="Historical Data",
        linewidth=2,
        color='#1f77b4'
    )
    
    # Plot median forecast
    median_forecast = np.median(forecast, axis=0)
    ax.plot(
        forecast_index,
        median_forecast,
        label="Median Forecast",
        linewidth=2,
        color='#ff7f0e'
    )
    
    # Add confidence interval
    ax.fill_between(
        forecast_index,
        np.percentile(forecast, 10, axis=0),
        np.percentile(forecast, 90, axis=0),
        alpha=0.3,
        interpolate=True,
        label="80% Confidence Interval",
        color='#ff7f0e'
    )
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Add title and labels
    plt.title(f'Future Market Cap Forecast: {item_name}', fontsize=14, pad=20)
    plt.ylabel('Market Cap (Billion USD)', fontsize=12)
    plt.xlabel('Date')
    
    # Improve the grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    
    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def main():
    # Convert test dataset to GluonTS format
    gluonts_test_dataset = convert_to_gluonts_dataset(test_dataset, freq)
    
    # Create test dataloader
    test_dataloader = create_test_dataloader(
        config=config,
        freq=freq,
        data=gluonts_test_dataset,
        batch_size=64,
    )
    
    # Generate forecasts
    print("Generating forecasts...")
    forecasts, _, _ = generate_forecasts(model, test_dataloader, device)
    print(f"Generated forecasts shape: {forecasts.shape}")
    
    # Plot forecasts for target companies
    for i, item in enumerate(test_dataset):
        if item["item_id"] in target_companies:
            print(f"Plotting forecast for {item['item_id']}...")
            plot_forecast(
                ts_data=item["target"],
                forecast=forecasts[i],
                item_name=item["item_id"],
                start_date=item["start"],
                freq=freq,
                prediction_length=prediction_length
            )

if __name__ == "__main__":
    main()