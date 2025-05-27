"""
`train.py`
Time Series Forecasting with Transformer Model
This script trains a Time Series Transformer model on market capitalization data.
"""

import os
import shutil
from functools import lru_cache, partial
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cached, Cyclic
from gluonts.time_feature import (
    get_lags_for_frequency,
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
from torch.optim import AdamW
from transformers import PretrainedConfig, TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import time
from datetime import timedelta


# ===== Helper Functions =====

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    """Convert a date to pandas Period with given frequency."""
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    """Transform start dates to pandas Periods."""
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch


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


def create_train_dataloader(
    config: PretrainedConfig,
    freq: str,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    """Create a dataloader for training data."""
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

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    instance_splitter = create_instance_splitter(config, "train")

    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def load_metadata(data_dir):
    """Load metadata from file."""
    metadata = {}
    with open(os.path.join(data_dir, "metadata.txt"), "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            metadata[key] = value
    return metadata


def forward_pass(model, batch, config, device):
    """Perform a forward pass through the model."""
    return model(
        static_categorical_features=batch["static_categorical_features"].to(device)
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"].to(device)
        if config.num_static_real_features > 0
        else None,
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        future_values=batch["future_values"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
        future_observed_mask=batch["future_observed_mask"].to(device),
    )


def save_model(model, model_path, config_path, freq, prediction_length, lags_sequence):
    """Save the model and its configuration."""
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    if os.path.exists(config_path):
        shutil.rmtree(config_path)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(config_path, exist_ok=True)

    # Save model state dictionary
    torch.save(model.state_dict(), os.path.join(model_path, "time_series_model.pth"))

    # Save the configuration
    model.config.to_json_file(os.path.join(config_path, "config.json"))

    # Save frequency and prediction length for later use
    with open(os.path.join(config_path, "metadata.txt"), "w") as f:
        f.write(f"freq={freq}\n")
        f.write(f"prediction_length={prediction_length}\n")
        f.write(f"lags_sequence={lags_sequence}\n")


def train_model(model, optimizer, train_dataloader, config, device, num_epochs=40):
    """Train the model for the specified number of epochs."""
    model.train()
    
    # Start total training timer
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        # Start epoch timer
        epoch_start_time = time.time()
        
        epoch_loss = 0
        num_batches = 0

        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = forward_pass(model, batch, config, device)
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item():.4f}")

        # Calculate epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed in {timedelta(seconds=epoch_duration)}. Average loss: {avg_epoch_loss:.4f}")
    
    # Calculate total training time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"Training completed in {timedelta(seconds=total_duration)}")
    print(f"Average time per epoch: {timedelta(seconds=total_duration/num_epochs)}")


def visualize_data(train_example, validation_example, test_example):
    """Visualize the train, validation, and test data."""
    figure, axes = plt.subplots(figsize=(12, 6))
    axes.plot(test_example["target"], color="red", label="Test")
    axes.plot(validation_example["target"], color="green", label="Validation")
    axes.plot(train_example["target"], color="blue", label="Train")
    axes.legend()
    axes.set_title("Time Series Data")
    axes.set_xlabel("Time Steps")
    axes.set_ylabel("Values")
    plt.tight_layout()
    plt.show()


# ===== Main Execution =====

# Load datasets
data_dir = "prepared_marketcap_dataset"
dataset = load_from_disk(f"{data_dir}/dataset")

print(f"Train dataset: {len(dataset['train'])} time series")
print(f"Validation dataset: {len(dataset['validation'])} time series")
print(f"Test dataset: {len(dataset['test'])} time series")

# Get examples from each split
train_example = dataset["train"][0]
validation_example = dataset["validation"][0]
test_example = dataset["test"][0]

# Load metadata
metadata = load_metadata(data_dir)
freq = metadata["freq"]
prediction_length = int(metadata["prediction_length"])

# Verify dataset structure
assert len(train_example["target"]) + prediction_length == len(validation_example["target"]), \
    "Validation example length doesn't match expected length"

# Visualize data
visualize_data(train_example, validation_example, test_example)

# Set up datasets
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Apply transformations
train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

# Get time features and lags
lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)

print(f"Lags sequence: {lags_sequence}")
print(f"Time features: {time_features}")

# Configure model
config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=prediction_length * 2,
    lags_sequence=lags_sequence,
    num_time_features=len(time_features) + 1,  # Add 1 for age feature
    num_static_categorical_features=3,
    cardinality = [
        int(metadata["cardinality_sector"]),
        int(metadata["cardinality_industry"]),
        int(metadata["cardinality_mcap"]),
    ],
    embedding_dimension=[4, 6, 3],   # Dimension of categorical embedding
    
    # transformer params:
    encoder_layers=4,
    decoder_layers=4,
    d_model=32,
)

model = TimeSeriesTransformerForPrediction(config)
print(f"Distribution output: {model.config.distribution_output}")

# Create dataloaders
train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

# Check batch shape
batch = next(iter(train_dataloader))
print("\nBatch shapes:")
for k, v in batch.items():
    print(f"{k}: {v.shape}, {v.type()}")

# Initialize accelerator and optimizer
global accelerator  # Make accelerator accessible to train_model function
accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Train the model
train_model(model, optimizer, train_dataloader, config, device, num_epochs=40)

# Save the model
unwrapped_model = accelerator.unwrap_model(model)
model_path = "models/marketcap_model_1000"
config_path = "models/marketcap_model_1000/config"
save_model(unwrapped_model, model_path, config_path, freq, prediction_length, lags_sequence)

print(f"Model saved to {model_path}")
print(f"Configuration saved to {config_path}")






















"""
`test_model.py`
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
            FieldName.FEAT_STATIC_CAT: item["feat_static_cat"],
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
    """Calculate MASE, sMAPE, and MAPE metrics for the forecasts."""
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")
    forecast_median = np.median(forecasts, axis=1)
    
    mase_metrics = []
    smape_metrics = []
    mape_metrics = []  # Added MAPE metrics list
    
    for item_id, ts in enumerate(test_dataset):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]
        
        # Calculate MASE
        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(training_data),
            periodicity=get_seasonality(freq)
        )
        mase_metrics.append(mase["mase"])
        
        # Calculate sMAPE
        smape = smape_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth)
        )
        smape_metrics.append(smape["smape"])
        
        # Calculate MAPE
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        mape = np.mean(np.abs((np.array(ground_truth) - forecast_median[item_id]) / 
                              (np.array(ground_truth) + epsilon))) * 100
        mape_metrics.append(mape)
        
    return mase_metrics, smape_metrics, mape_metrics


def plot_metrics_relationship(mase_metrics, smape_metrics, xlabel, ylabel):
    """Plot the relationship between MASE and sMAPE metrics."""
    plt.figure(figsize=(10, 6))
    plt.scatter(mase_metrics, smape_metrics, alpha=0.3, color='#1f77b4', s=50)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"Relationship between {xlabel} and {ylabel} metrics", fontsize=14)
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

# Load model and configuration
model_dir = "../model"
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
data_dir = "../prepared_marketcap_dataset"
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
mase_metrics, smape_metrics, mape_metrics = calculate_metrics(forecasts, test_dataset, prediction_length, freq)
print(f"MASE: {np.mean(mase_metrics):.4f}")
print(f"sMAPE: {np.mean(smape_metrics):.4f}")
print(f"MAPE: {np.mean(mape_metrics):.4f}%")

# Visualize results
plot_metrics_relationship(mase_metrics, smape_metrics, "MASE", "sMAPE")
plot_metrics_relationship(mase_metrics, mape_metrics, "MASE", "MAPE")

# Plot forecasts for selected companies
target_companies = {"Apple", "Microsoft", "NVIDIA", "Tencent", "ICBC", "Alibaba"}
print(f"Plotting forecasts for selected companies: {', '.join(target_companies)}")
plot_selected_forecasts(test_dataset, forecasts, prediction_length, freq, target_companies)


















"""
`forecast_future.py`
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
MODEL_DIR = "../model"  # Update with your model directory
DATA_DIR = "../prepared_marketcap_dataset"
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
            FieldName.FEAT_STATIC_CAT: item["feat_static_cat"],
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
    
    # IMPROVED X-AXIS FORMATTING - Multiple approaches:
    
    # Calculate total time span to choose appropriate formatting
    total_dates = len(historical_index) + len(forecast_index)
    date_range = (forecast_index[-1] - historical_index[0]).days
    
    # Adaptive date formatting based on data span
    if date_range <= 365:  # Less than 1 year
        # Show every 2-3 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    elif date_range <= 1095:  # 1-3 years
        # Show every 6 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    elif date_range <= 1825:  # 3-5 years
        # Show yearly
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    else:  # More than 5 years
        # Show every 2 years
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Alternative: Limit maximum number of ticks
    # ax.xaxis.set_major_locator(mdates.MaxNLocator(nbins=8))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Rotate labels and improve spacing
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
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
    
    # Automatic date formatting for better readability
    fig.autofmt_xdate()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    #os.makedirs('forecasts/transformers_forecasts', exist_ok=True)
    #plt.savefig(f'forecasts/transformers_forecasts/{item_name}_forecast.png', 
    #            dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

# ═════════════════════════════════════════════════════════════════
# Main Execution
# ═════════════════════════════════════════════════════════════════

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
