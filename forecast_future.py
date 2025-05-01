"""
Market Cap Forecasting Tool
==========================
This module predicts future market capitalization for companies using 
a pre-trained time series transformer model.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Dict, Any, Tuple

# HuggingFace & GluonTS imports
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction, PretrainedConfig
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

# ═════════════════════════════════════════════════════════════════
# Configuration and Setup
# ═════════════════════════════════════════════════════════════════

# System paths and constants
MODEL_DIR = "models/marketcap_model_1000"  # Update with your model directory
DATA_DIR = "prepared_marketcap_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target companies for visualization
TARGET_COMPANIES = {
    "Apple", "Microsoft", "NVIDIA", 
    "Tencent", "ICBC", "Alibaba"
}

# ═════════════════════════════════════════════════════════════════
# Data Processing Functions
# ═════════════════════════════════════════════════════════════════

def load_model_and_config() -> Tuple[TimeSeriesTransformerForPrediction, 
                                     TimeSeriesTransformerConfig, 
                                     Dict[str, str]]:
    """
    Load the model, configuration, and metadata.
    
    Returns:
        Tuple containing:
        - The loaded model
        - Model configuration
        - Metadata dictionary
    """
    # Load model configuration
    config_path = os.path.join(MODEL_DIR, "config")
    config = TimeSeriesTransformerConfig.from_pretrained(config_path)
    
    # Load metadata
    metadata = {}
    with open(os.path.join(config_path, "metadata.txt"), "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            metadata[key] = value
    
    # Initialize and load model
    model = TimeSeriesTransformerForPrediction(config)
    model_path = os.path.join(MODEL_DIR, "time_series_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, config, metadata


def create_transformation(freq: str, config: PretrainedConfig) -> Chain:
    """
    Create the transformation pipeline for time series data.
    
    Args:
        freq: Time series frequency string (e.g., 'D', 'M', 'Q')
        config: Model configuration
        
    Returns:
        Chain of transformations to apply to the dataset
    """
    # Determine which fields to remove based on features used
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # Build transformation chain
    transformation_chain = [RemoveFields(field_names=remove_field_names)]
    
    # Add static categorical feature processing if needed
    if config.num_static_categorical_features > 0:
        transformation_chain.append(
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            )
        )
    
    # Add static real feature processing if needed
    if config.num_static_real_features > 0:
        transformation_chain.append(
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            )
        )
    
    # Add core transformations
    transformation_chain.extend([
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
    ])
    
    # Stack all time features
    time_features = [FieldName.FEAT_TIME, FieldName.FEAT_AGE]
    if config.num_dynamic_real_features > 0:
        time_features.append(FieldName.FEAT_DYNAMIC_REAL)
        
    transformation_chain.append(
        VstackFeatures(
            output_field=FieldName.FEAT_TIME,
            input_fields=time_features,
        )
    )
    
    # Rename fields to match model expectations
    transformation_chain.append(
        RenameFields(
            mapping={
                FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                FieldName.FEAT_STATIC_REAL: "static_real_features",
                FieldName.FEAT_TIME: "time_features",
                FieldName.TARGET: "values",
                FieldName.OBSERVED_VALUES: "observed_mask",
            }
        )
    )
    
    return Chain(transformation_chain)


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> InstanceSplitter:
    """
    Create an instance splitter based on the specified mode.
    
    Args:
        config: Model configuration
        mode: One of 'train', 'validation', or 'test'
        train_sampler: Sampler to use in training mode
        validation_sampler: Sampler to use in validation mode
        
    Returns:
        Configured InstanceSplitter
    """
    assert mode in ["train", "validation", "test"], f"Invalid mode: {mode}"

    # Select appropriate sampler based on mode
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


def convert_to_gluonts_dataset(hf_dataset, freq: str) -> ListDataset:
    """
    Convert HuggingFace dataset format to GluonTS ListDataset format.
    
    Args:
        hf_dataset: HuggingFace dataset
        freq: Time series frequency string
        
    Returns:
        GluonTS ListDataset
    """
    data = []
    for item in hf_dataset:
        data.append({
            FieldName.START: pd.Period(item["start"], freq=freq),
            FieldName.TARGET: item["target"],
            FieldName.FEAT_STATIC_CAT: [item["feat_static_cat"][0]],
            FieldName.ITEM_ID: item["item_id"]
        })
    return ListDataset(data, freq=freq)


def create_test_dataloader(
    config: PretrainedConfig,
    freq: str,
    data: ListDataset,
    batch_size: int = 64,
) -> Any:
    """
    Create a dataloader for generating forecasts from the end of time series.
    
    Args:
        config: Model configuration
        freq: Time series frequency
        data: GluonTS dataset
        batch_size: Batch size for inference
        
    Returns:
        DataLoader for test data
    """
    # Define which fields to include in prediction input
    prediction_input_names = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    
    if config.num_static_categorical_features > 0:
        prediction_input_names.append("static_categorical_features")

    if config.num_static_real_features > 0:
        prediction_input_names.append("static_real_features")

    # Apply transformations
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # Create test instance splitter
    instance_sampler = create_instance_splitter(config, "test")
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    # Return stacked batch iterator
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=prediction_input_names,
    )

# ═════════════════════════════════════════════════════════════════
# Forecasting Functions
# ═════════════════════════════════════════════════════════════════

def generate_forecasts(
    model: TimeSeriesTransformerForPrediction, 
    dataloader: Any, 
    device: torch.device,
    config: PretrainedConfig
) -> Tuple[np.ndarray, List, List]:
    """
    Generate forecasts using the model.
    
    Args:
        model: The prediction model
        dataloader: DataLoader containing test data
        device: Device to run inference on
        config: Model configuration
        
    Returns:
        Tuple containing:
        - Forecast arrays
        - Item IDs (if available)
        - Forecast start dates (if available)
    """
    model.eval()
    forecasts = []
    item_ids = []
    forecast_start_dates = []

    print(f"Running inference on {device}...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}...")
                
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
            
            # Extract metadata if available
            if hasattr(batch, "item_id"):
                item_ids.extend(batch.item_id)
            if hasattr(batch, FieldName.FORECAST_START):
                forecast_start_dates.extend(batch[FieldName.FORECAST_START])
    
    # Stack all forecasts if any were generated
    if forecasts:
        forecasts = np.vstack(forecasts)
    
    return forecasts, item_ids, forecast_start_dates


def plot_forecast(
    ts_data: np.ndarray, 
    forecast: np.ndarray, 
    item_name: str, 
    start_date: str, 
    freq: str, 
    prediction_length: int
) -> None:
    """
    Plot the forecast along with historical data.
    
    Args:
        ts_data: Historical time series data
        forecast: Model's forecast samples
        item_name: Name of the time series (e.g., company name)
        start_date: Start date of the time series
        freq: Time series frequency
        prediction_length: Length of forecast period
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
        linewidth=2.5,
        color='#1f77b4'
    )
    
    # Plot median forecast
    median_forecast = np.median(forecast, axis=0)
    ax.plot(
        forecast_index,
        median_forecast,
        label="Median Forecast",
        linewidth=2.5,
        color='#ff7f0e'
    )
    
    # Add confidence intervals
    ax.fill_between(
        forecast_index,
        np.percentile(forecast, 10, axis=0),
        np.percentile(forecast, 90, axis=0),
        alpha=0.3,
        interpolate=True,
        label="80% Confidence Interval",
        color='#ff7f0e'
    )
    
    # Add more extreme confidence interval
    ax.fill_between(
        forecast_index,
        np.percentile(forecast, 2.5, axis=0),
        np.percentile(forecast, 97.5, axis=0),
        alpha=0.15,
        interpolate=True,
        label="95% Confidence Interval",
        color='#ff7f0e'
    )
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Add title and labels
    plt.title(f'Market Cap Forecast: {item_name}', fontsize=16, pad=20)
    plt.ylabel('Market Cap (Billion USD)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    
    # Add a vertical line to separate historical data from forecasts
    ax.axvline(x=historical_index[-1], color='gray', linestyle='--', alpha=0.7)
    ax.text(historical_index[-1], ax.get_ylim()[1]*0.95, 'Forecast Start', 
            horizontalalignment='center', color='gray')
    
    # Improve the grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    
    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate()
    
    # Add watermark
    fig.text(0.99, 0.01, 'Generated with TimeSeriesTransformer', 
             fontsize=8, color='gray', ha='right', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('forecasts/transformers_forecasts', exist_ok=True)
    plt.savefig(f'forecasts/transformers_forecasts/{item_name}_forecast.png', dpi=300, bbox_inches='tight')
    
    plt.show()

# ═════════════════════════════════════════════════════════════════
# Main Function
# ═════════════════════════════════════════════════════════════════

def main():
    """Run the forecasting pipeline."""
    print(f"Running on device: {DEVICE}")
    
    # Load model, configuration, and metadata
    model, config, metadata = load_model_and_config()
    
    # Extract key parameters from metadata
    freq = metadata["freq"]
    prediction_length = int(metadata["prediction_length"])
    
    print("Model loaded successfully.")
    print(f"Frequency: {freq}, Prediction length: {prediction_length}")
    
    # Load dataset
    dataset = load_from_disk(f"{DATA_DIR}/dataset")
    test_dataset = dataset["test"]
    print(f"Test dataset loaded with {len(test_dataset)} time series.")
    
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
    forecasts, _, _ = generate_forecasts(model, test_dataloader, DEVICE, config)
    print(f"Generated forecasts shape: {forecasts.shape}")
    
    # Plot forecasts for target companies
    found_companies = 0
    for i, item in enumerate(test_dataset):
        if item["item_id"] in TARGET_COMPANIES:
            found_companies += 1
            print(f"Plotting forecast for {item['item_id']}...")
            plot_forecast(
                ts_data=item["target"],
                forecast=forecasts[i],
                item_name=item["item_id"],
                start_date=item["start"],
                freq=freq,
                prediction_length=prediction_length
            )
    
    print(f"Forecasting complete. Generated plots for {found_companies} target companies.")

if __name__ == "__main__":
    main()