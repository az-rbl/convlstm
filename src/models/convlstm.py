"""Deep ConvLSTM architecture with Change Detection and 128 -> 16 -> 1 FC head."""

from typing import List, Optional, Tuple
import keras
from keras import layers
import tensorflow as tf

from src.config import AppConfig, ModelConfig


def build_convlstm_model(
    input_shape: Tuple[Optional[int], int, int, int] = (16, 112, 112, 3),
    depth: int = 9,
    filters: Optional[List[int]] = None,
    kernel_sizes: Optional[List[int]] = None,
    dense_units: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.1,
    use_batch_norm: bool = True,
    name: str = "DeepConvLSTM_Classifier",
) -> keras.Model:
    """Builds a Deep ConvLSTM classification model for video frame differences.

    Architecture:
      1. Input: (Batch, Time, Height, Width, Channels=3) frame differences.
      2. Backbone: 9 stacked ConvLSTM2D layers (depth=9).
         - Intermediate ConvLSTM layers return sequences (return_sequences=True).
         - The final ConvLSTM layer returns the last temporal state (return_sequences=False).
         - Interleaved BatchNormalization and spatial pooling for gradient stability and receptive field.
      3. Pooling: GlobalAveragePooling2D reducing 2D spatial feature maps.
      4. Fully Connected Classification Head:
         - Dense(128) -> ReLU -> [BatchNorm] -> Dropout
         - Dense(16)  -> ReLU -> [BatchNorm] -> Dropout
         - Dense(1)   -> Sigmoid (Probability of violence / fight)

    Args:
        input_shape: (sequence_length, height, width, channels)
        depth: Number of stacked ConvLSTM layers (default: 9).
        filters: List of filter counts for each ConvLSTM layer.
        kernel_sizes: List of kernel sizes for each ConvLSTM layer.
        dense_units: Units for classification head (default: [128, 16, 1]).
        dropout_rate: Dropout rate for dense layers.
        recurrent_dropout: Recurrent dropout for ConvLSTM cells.
        use_batch_norm: Whether to apply BatchNormalization layers.
        name: Name of the model.

    Returns:
        keras.Model instance.
    """
    if dense_units is None:
        dense_units = [128, 16, 1]

    if filters is None:
        # Default 9-layer filter progression
        filters = [32, 32, 32, 64, 64, 64, 128, 128, 128]

    if kernel_sizes is None:
        kernel_sizes = [3] * depth

    if len(filters) < depth:
        # Extend filters if depth is larger than provided list
        filters = filters + [filters[-1]] * (depth - len(filters))
    if len(kernel_sizes) < depth:
        kernel_sizes = kernel_sizes + [3] * (depth - len(kernel_sizes))

    inputs = layers.Input(shape=input_shape, name="difference_frames_input")
    x = inputs

    # Build depth-9 ConvLSTM layers
    for layer_idx in range(depth):
        is_last = layer_idx == (depth - 1)
        num_filter = filters[layer_idx]
        k_size = (kernel_sizes[layer_idx], kernel_sizes[layer_idx])

        x = layers.ConvLSTM2D(
            filters=num_filter,
            kernel_size=k_size,
            padding="same",
            return_sequences=not is_last,
            recurrent_dropout=recurrent_dropout,
            name=f"convlstm_depth_{layer_idx + 1}",
        )(x)

        if use_batch_norm:
            # BatchNormalization across features
            x = layers.BatchNormalization(name=f"bn_depth_{layer_idx + 1}")(x)

        # Interleave spatial downsampling at layer 3 and 6 to enlarge receptive field
        # and prevent GPU memory explosion while preserving depth 9
        if (layer_idx + 1) in [3, 6] and not is_last:
            x = layers.TimeDistributed(
                layers.MaxPooling2D(pool_size=(2, 2)),
                name=f"pool_depth_{layer_idx + 1}",
            )(x)

    # Spatial reduction: Global Average Pooling 2D
    # At this point, x has shape (Batch, H_final, W_final, Filters_last)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Fully connected layers: 128 -> 16 -> 1
    # Dense 128
    x = layers.Dense(dense_units[0], activation="relu", name="dense_128")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_dense_128")(x)
    x = layers.Dropout(dropout_rate, name="dropout_128")(x)

    # Dense 16
    x = layers.Dense(dense_units[1], activation="relu", name="dense_16")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_dense_16")(x)
    x = layers.Dropout(dropout_rate, name="dropout_16")(x)

    # Dense 1 (Output)
    outputs = layers.Dense(dense_units[2], activation="sigmoid", name="output_dense_1")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_model_from_config(config: AppConfig) -> keras.Model:
    """Builds model using AppConfig dataclass."""
    data_cfg = config.data
    model_cfg = config.model

    input_shape = (
        data_cfg.sequence_length,
        data_cfg.image_size[0],
        data_cfg.image_size[1],
        data_cfg.channels,
    )

    return build_convlstm_model(
        input_shape=input_shape,
        depth=model_cfg.depth,
        filters=model_cfg.convlstm_filters,
        kernel_sizes=model_cfg.kernel_sizes,
        dense_units=model_cfg.dense_units,
        dropout_rate=model_cfg.dropout_rate,
        recurrent_dropout=model_cfg.recurrent_dropout,
        use_batch_norm=model_cfg.use_batch_norm,
        name=model_cfg.name,
    )
