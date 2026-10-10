"""Change detection pipeline for video sequences.

Computes inter-frame consecutive differences preserving signs and motion vectors
across all three color channels.
"""

from typing import Union
import numpy as np
import tensorflow as tf


def compute_frame_differences_numpy(
    frames: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Computes signed difference between consecutive frames using NumPy.

    Args:
        frames: Video frames array of shape (T, H, W, 3) with dtype uint8 or float32.
        normalize: If True, normalizes pixel values to [-1.0, 1.0].

    Returns:
        differences: Array of shape (T - 1, H, W, 3) representing Delta I_t = I_{t+1} - I_t.
    """
    if frames.ndim != 4:
        raise ValueError(
            f"Expected 4D array of shape (T, H, W, C), got ndim={frames.ndim}"
        )
    if frames.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 frames to compute differences, got {frames.shape[0]}"
        )

    # Cast to float32 to prevent unsigned underflow
    float_frames = frames.astype(np.float32)
    if normalize and float_frames.max() > 1.0:
        float_frames = float_frames / 255.0

    # Consecutive difference: I_{t+1} - I_t
    diffs = float_frames[1:] - float_frames[:-1]
    return diffs.astype(np.float32)


@tf.function
def compute_frame_differences_tf(
    frames: tf.Tensor,
    normalize: bool = True,
) -> tf.Tensor:
    """Computes signed difference between consecutive frames using TensorFlow ops.

    Args:
        frames: Video tensor of shape (batch_or_T, T, H, W, 3) or (T, H, W, 3).
        normalize: If True and input max > 1.0, normalizes by 255.0.

    Returns:
        diffs: Consecutive differences along the temporal axis.
    """
    float_frames = tf.cast(frames, tf.float32)
    if normalize:
        float_frames = tf.clip_by_value(float_frames / 255.0, 0.0, 1.0)

    # Assumes shape (T, H, W, C)
    diffs = float_frames[1:] - float_frames[:-1]
    return diffs
