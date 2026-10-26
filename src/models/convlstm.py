"""ConvLSTM architecture with Change Detection: Single ConvLSTM layer (9 filters) and 128 -> 16 -> 1 FC head."""

from typing import List, Optional, Tuple, Union
import keras
from keras import layers
import tensorflow as tf

from src.config import AppConfig


def build_convlstm_model(
    input_shape: Tuple[Optional[int], int, int, int] = (16, 112, 112, 3),
    filters: int = 9,
    depth_multiplier: int = 2,
    kernel_size: Union[Tuple[int, int], List[int]] = (3, 3),
    dense_units: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.1,
    use_batch_norm: bool = True,
    name: str = "ConvLSTM_Classifier",
) -> keras.Model:
    """Builds a lightweight ConvLSTM classification model for video frame differences.

    Architecture:
      1. Input: (Batch, Time, Height, Width, Channels=3) consecutive frame differences.
      2. Backbone: Single ConvLSTM2D layer with 9 filters (return_sequences=False).
         - Extracts spatiotemporal motion features across the difference sequence.
         - Output shape: (Batch, Height, Width, 9).
      3. Pooling: GlobalAveragePooling2D reducing 2D spatial feature map to (Batch, 9).
      4. Fully Connected Classification Head:
         - Dense(128) -> ReLU -> [BatchNorm] -> Dropout
         - Dense(16)  -> ReLU -> [BatchNorm] -> Dropout
         - Dense(1)   -> Sigmoid (Probability of violence / fight)

    Args:
        input_shape: (sequence_length, height, width, channels)
        filters: Number of filters in the single ConvLSTM layer (default: 9).
        kernel_size: Convolution kernel size (default: (3, 3)).
        dense_units: Units for classification head (default: [128, 16, 1]).
        dropout_rate: Dropout rate for dense layers.
        recurrent_dropout: Recurrent dropout for ConvLSTM cell.
        use_batch_norm: Whether to apply BatchNormalization layers.
        name: Name of the model.

    Returns:
        keras.Model instance.
    """
    if dense_units is None:
        dense_units = [128, 16, 1]

    if isinstance(kernel_size, list):
        k_size = tuple(kernel_size)
    else:
        k_size = kernel_size

    inputs = layers.Input(shape=input_shape, name="difference_frames_input")

    # 1. Single ConvLSTM2D layer with 9 filters
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=k_size,
        padding="same",
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
        name="convlstm_layer",
    )(inputs)

    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_convlstm")(x)

    # 2. Global Average Pooling (Height, Width) -> 1D vector (9,)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ==========================================
    # Post-processing: Depthwise Convolution
    # ==========================================
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depth_multiplier=depth_multiplier,
        padding="same",
        activation="relu",
        name="depthwise_conv_post",
    )(x)
    # 3. Fully connected layers: 128 -> 16 -> 1
    # Dense 128
    x = layers.Dense(dense_units[0], activation="relu", name="dense_128")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_dense_128")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_128")(x)

    # Dense 16
    x = layers.Dense(dense_units[1], activation="relu", name="dense_16")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_dense_16")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_16")(x)

    # Dense 1 (Output probability)
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
        filters=model_cfg.filters,
        kernel_size=model_cfg.kernel_size,
        dense_units=model_cfg.dense_units,
        dropout_rate=model_cfg.dropout_rate,
        recurrent_dropout=model_cfg.recurrent_dropout,
        use_batch_norm=model_cfg.use_batch_norm,
        name=model_cfg.name,
    )
