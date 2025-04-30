"""
Market Capitalization Time Series Forecasting - Model Testing
This script loads a trained Time Series Transformer model and evaluates its performance on test data.
"""

import os
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from evaluate import load
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.time_feature import (
    get_seasonality,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    RenameFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from transformers import PretrainedConfig, TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


# ===== Helper Functions =====

def load_metadata(config_path: str) -> dict:
    """Load metadata from file."""
    metadata = {}
    with open(os.path.join(config_path, "metadata.txt"), "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            metadata[key] = value
    return metadata


def convert_to_gluonts_dataset(hf_dataset, freq: str) -> ListDataset:
    """Convert HuggingFace dataset to GluonTS ListDataset format."""
    data = []
    for item in hf_dataset:
        data.append({
            FieldName.START: pd.Period(item["start"], freq=freq),
            FieldName.TARGET: item["target"],
            FieldName.FEAT_STATIC_CAT: [item["feat_static_cat"][0]],
            # Uncomment if needed:
            # FieldName.FEAT_STATIC_REAL: item["feat_static_real"],
            FieldName.ITEM_ID: item["item_id"]
        })
    return ListDataset(data, freq=freq)


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    """Create a transformation pipeline for data preprocessing."""
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        # Step 1: Remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # Step 2: Convert the data to NumPy
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
            # Step 3: Handle NaN values - fill target with zeros and create mask
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Step 4: Add temporal features based on frequency
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # Step 5: Add age feature (time series position indicator)
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # Step 6: Stack temporal features
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # Step 7: Rename fields to match HuggingFace names
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
) -> Transformation:
    """Create an instance splitter for the specified mode."""
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
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


def create_backtest_dataloader(
    config: PretrainedConfig,
    freq: str,
    data,
    batch_size: int,
    **kwargs,
):
    """Create a dataloader for backtesting."""
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
    transformed_data = transformation.apply(data)

    # Use validation instance splitter to sample the last context window
    instance_sampler = create_instance_splitter(config, "validation")

    # Apply transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


def generate_forecasts(model, test_dataloader, config, device):
    """Generate forecasts using the model."""
    model.eval()
    forecasts = []

    with torch.no_grad():
        for batch in test_dataloader:
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
            forecasts.append(outputs.sequences.cpu().numpy())

    return np.vstack(forecasts)


def calculate_metrics(forecasts, test_dataset, prediction_length, freq):
    """Calculate MASE and sMAPE metrics for the forecasts."""
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, axis=1)
    
    mase_metrics = []
    smape_metrics = []

    for item_id, ts in enumerate(test_dataset):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]

        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(training_data),
            periodicity=get_seasonality(freq)
        )
        mase_metrics.append(mase["mase"])

        smape = smape_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth)
        )
        smape_metrics.append(smape["smape"])

    return mase_metrics, smape_metrics


def plot_metrics_relationship(mase_metrics, smape_metrics):
    """Plot the relationship between MASE and sMAPE metrics."""
    plt.figure(figsize=(10, 6))
    plt.scatter(mase_metrics, smape_metrics, alpha=0.3, color='#1f77b4', s=50)
    plt.xlabel("MASE", fontsize=12)
    plt.ylabel("sMAPE", fontsize=12)
    plt.title("Relationship between MASE and sMAPE metrics", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_forecast(ts_index, test_dataset, forecasts, prediction_length, freq):
    """Plot the forecast for a specific time series."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create date range for x-axis
    index = pd.period_range(
        start=test_dataset[ts_index][FieldName.START],
        periods=len(test_dataset[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # Configure x-axis ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Plot historical data
    ax.plot(
        index[-2 * prediction_length:],
        test_dataset[ts_index]["target"][-2 * prediction_length:],
        label="Historical Data",
        linewidth=2.5,
        color='#1f77b4'
    )

    # Plot median forecast
    ax.plot(
        index[-prediction_length:],
        np.median(forecasts[ts_index], axis=0),
        label="Median Forecast",
        linewidth=2.5,
        color='#ff7f0e'
    )

    # Add confidence interval
    ax.fill_between(
        index[-prediction_length:],
        forecasts[ts_index].mean(0) - forecasts[ts_index].std(axis=0),
        forecasts[ts_index].mean(0) + forecasts[ts_index].std(axis=0),
        alpha=0.3,
        interpolate=True,
        label="±1 Std Dev",
        color='#ff7f0e'
    )

    # Customize plot appearance
    title = f'Market Cap Forecast: {test_dataset[ts_index]["item_id"]}'
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('Market Cap (Billion USD)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_selected_forecasts(test_dataset, forecasts, prediction_length, freq, target_items=None):
    """Plot forecasts for selected companies."""
    if target_items is None:
        target_items = {"Apple", "Microsoft", "NVIDIA", "Tencent", "ICBC", "Alibaba"}
    
    for i, item in enumerate(test_dataset):
        if item["item_id"] in target_items:
            plot_forecast(i, test_dataset, forecasts, prediction_length, freq)


# ===== Main Execution =====

def main():
    # Load model and configuration
    model_dir = "marketcap_model_900_12"
    model_path = os.path.join(model_dir, "time_series_model.pth")
    config_path = os.path.join(model_dir, "config")
    
    # Load configuration and metadata
    config = TimeSeriesTransformerConfig.from_pretrained(config_path)
    metadata = load_metadata(config_path)
    
    freq = metadata["freq"]
    prediction_length = int(metadata["prediction_length"])
    
    print(f"Model configuration loaded from {config_path}")
    print(f"Frequency: {freq}")
    print(f"Prediction length: {prediction_length}")
    
    # Initialize and load model
    model = TimeSeriesTransformerForPrediction(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Load test dataset
    data_dir = "prepared_marketcap_dataset"
    dataset = load_from_disk(f"{data_dir}/dataset")
    test_dataset = dataset["test"]
    
    print(f"Test dataset loaded: {len(test_dataset)} time series")
    
    # Convert to GluonTS format
    gluonts_test_dataset = convert_to_gluonts_dataset(test_dataset, freq)
    
    # Create dataloader
    test_dataloader = create_backtest_dataloader(
        config=config,
        freq=freq,
        data=gluonts_test_dataset,
        batch_size=64,
    )
    
    # Generate forecasts
    print("Generating forecasts...")
    forecasts = generate_forecasts(model, test_dataloader, config, device)
    print(f"Generated forecasts shape: {forecasts.shape}")
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    mase_metrics, smape_metrics = calculate_metrics(forecasts, test_dataset, prediction_length, freq)
    print(f"MASE: {np.mean(mase_metrics):.4f}")
    print(f"sMAPE: {np.mean(smape_metrics):.4f}")
    
    # Visualize results
    plot_metrics_relationship(mase_metrics, smape_metrics)
    
    # Plot forecasts for selected companies
    target_companies = {"Apple", "Microsoft", "NVIDIA", "Tencent", "ICBC", "Alibaba"}
    print(f"Plotting forecasts for selected companies: {', '.join(target_companies)}")
    plot_selected_forecasts(test_dataset, forecasts, prediction_length, freq, target_companies)


if __name__ == "__main__":
    main()