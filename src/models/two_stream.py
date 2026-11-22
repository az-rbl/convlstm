"""Two-stream converging architecture for video classification (Garcia-Cobo & SanMiguel, CVIU 2023).

Combines:
  - Pose stream (OpenPose skeletons on black background + 9-filter Conv2D)
  - Motion stream (Consecutive frame differences + 9-filter ConvLSTM2D)
  - Pointwise Addition Fusion: BatchNorm(F_RGB) + tanh(F_Change)
  - Spatiotemporal Aggregator (32-filter ConvLSTM2D)
  - Depthwise post-processing (DepthwiseConv2D)
  - 128 -> 16 -> 1 Fully Connected classification head
"""

from typing import List, Optional, Tuple
import keras
from keras import layers
import tensorflow as tf

from src.config import AppConfig


def build_two_stream_model(
    input_shape: Tuple[Optional[int], int, int, int] = (51, 100, 100, 3),
    conv_filters: int = 9,
    convlstm_diff_filters: int = 9,
    aggregator_filters: int = 32,
    depth_multiplier: int = 2,
    padding: str = "valid",
    dense_units: Optional[List[int]] = None,
    dropout_rate: float = 0.0,
    name: str = "TwoStream_ConvLSTM_Classifier",
) -> keras.Model:
    """Builds the complete Two-Stream converging video classifier (Garcia-Cobo & SanMiguel, CVIU 2023).

    Args:
        input_shape: (frames_per_video, height, width, channels) where frames_per_video = T + 1 (e.g. 51).
        conv_filters: Number of filters in the pose Conv2D layer (default: 9).
        convlstm_diff_filters: Number of filters in the difference ConvLSTM2D (default: 9).
        aggregator_filters: Number of filters in temporal aggregator ConvLSTM2D (default: 32).
        depth_multiplier: Multiplier for DepthwiseConv2D post-processing (default: 2 -> 64 filters).
        padding: Padding mode for convolution and ConvLSTM layers (default: 'valid' matching CVIU 2023).
        dense_units: Fully connected classification units (default: [128, 16, 1]).
        dropout_rate: Dropout rate for dense layers (default: 0.0 matching CVIU 2023).
        name: Model name.

    Returns:
        keras.Model instance accepting [inputs_raw, inputs_openpose].
    """
    if dense_units is None:
        dense_units = [128, 16, 1]

    # Two input streams: raw video and skeleton video
    inputs_raw = layers.Input(shape=input_shape, name="raw_video_input")
    inputs_openpose = layers.Input(shape=input_shape, name="openpose_skeleton_input")

    # ==========================================
    # Stream 1: Change Detection Pipeline
    # ==========================================
    # Compute consecutive frame differences: I_{t+1} - I_t
    inputs_diff = layers.Lambda(
        lambda video: video[:, 1:] - video[:, :-1],
        name="frame_difference_lambda",
    )(inputs_raw)

    inputs_diff_norm = layers.BatchNormalization(name="bn_difference")(inputs_diff)

    inputs_diff_time_info_weight = layers.ConvLSTM2D(
        filters=convlstm_diff_filters,
        kernel_size=(3, 3),
        padding=padding,
        return_sequences=True,
        activation="tanh",
        name="convlstm_diff_stream",
    )(inputs_diff_norm)

    # ==========================================
    # Stream 2: RGB / OpenPose Skeleton Pipeline
    # ==========================================
    # Match temporal length T = frames_per_video - 1
    inputs_to_weight = layers.Lambda(
        lambda skel: skel[:, :-1],
        name="skeleton_slice_lambda",
    )(inputs_openpose)

    # 9-filter convolutional layer applied to each skeleton frame
    conv_pose_layer = layers.Conv2D(
        filters=conv_filters,
        kernel_size=(3, 3),
        padding=padding,
        activation="relu",
        name="conv2d_pose_layer",
    )
    inputs_openpose_soft = layers.TimeDistributed(
        conv_pose_layer,
        name="td_conv2d_pose",
    )(inputs_to_weight)

    # Centered Batch Normalization (scale=False, center=False)
    inputs_openpose_norm = layers.BatchNormalization(
        scale=False,
        center=False,
        name="bn_pose_centered",
    )(inputs_openpose_soft)

    # ==========================================
    # Fusion: Pointwise Addition (Eq. 5)
    # F_combined = BatchNorm(F_RGB) + tanh(F_Change)
    # ==========================================
    inputs_weighted = layers.Add(name="addition_fusion")(
        [inputs_openpose_norm, inputs_diff_time_info_weight]
    )

    # ==========================================
    # Spatiotemporal Aggregator: ConvLSTM (32 filters)
    # ==========================================
    x = layers.ConvLSTM2D(
        filters=aggregator_filters,
        kernel_size=(3, 3),
        padding=padding,
        return_sequences=False,
        activation="tanh",
        name="convlstm_aggregator",
    )(inputs_weighted)

    # ==========================================
    # Post-processing: Depthwise Convolution
    # ==========================================
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depth_multiplier=depth_multiplier,
        padding=padding,
        activation="relu",
        name="depthwise_conv_post",
    )(x)

    # ==========================================
    # Classification Head: GAP + 128 -> 16 -> 1
    # ==========================================
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(dense_units[0], activation="relu", name="dense_128")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_128")(x)

    x = layers.Dense(dense_units[1], activation="relu", name="dense_16")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="dropout_16")(x)

    outputs = layers.Dense(dense_units[2], activation="sigmoid", name="output_dense_1")(x)

    model = keras.Model(
        inputs=[inputs_raw, inputs_openpose],
        outputs=outputs,
        name=name,
    )
    return model


def build_two_stream_from_config(config: AppConfig) -> keras.Model:
    """Builds two-stream model from AppConfig."""
    data_cfg = config.data
    model_cfg = config.model

    input_shape = (
        data_cfg.sequence_length + 1,
        data_cfg.image_size[0],
        data_cfg.image_size[1],
        data_cfg.channels,
    )

    return build_two_stream_model(
        input_shape=input_shape,
        conv_filters=getattr(model_cfg, "conv_filters", 9),
        convlstm_diff_filters=getattr(model_cfg, "convlstm_diff_filters", 9),
        aggregator_filters=getattr(model_cfg, "aggregator_filters", 32),
        depth_multiplier=getattr(model_cfg, "depth_multiplier", 2),
        padding=getattr(model_cfg, "padding", "valid"),
        dense_units=model_cfg.dense_units,
        dropout_rate=model_cfg.dropout_rate,
        name=model_cfg.name,
    )
