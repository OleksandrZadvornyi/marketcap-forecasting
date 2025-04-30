"""
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
    for epoch in range(num_epochs):
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

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")


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

def main():
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
        num_static_categorical_features=1,
        cardinality=[len(train_dataset)],  # Number of unique time series
        embedding_dimension=[2],  # Dimension of categorical embedding
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
    model_path = "marketcap_model"
    config_path = "marketcap_model/config"
    save_model(unwrapped_model, model_path, config_path, freq, prediction_length, lags_sequence)
    
    print(f"Model saved to {model_path}")
    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    main()