"""Unit tests for Change Detection pipeline and ConvLSTM (single layer, 9 filters) model."""

import numpy as np
import pytest
import tensorflow as tf

from src.data.change_detection import compute_frame_differences_numpy
from src.models.convlstm import build_convlstm_model


def test_frame_differences_shape_and_range():
    # Synthetic video of 10 frames with (32, 32, 3)
    raw_frames = np.random.randint(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    diffs = compute_frame_differences_numpy(raw_frames, normalize=True)

    # Expected shape: (9, 32, 32, 3)
    assert diffs.shape == (9, 32, 32, 3)
    assert diffs.dtype == np.float32
    # Differences must lie in [-1.0, 1.0]
    assert np.all(diffs >= -1.0) and np.all(diffs <= 1.0)


def test_single_convlstm_9_filters_architecture():
    # Build single ConvLSTM model with 9 filters
    model = build_convlstm_model(
        input_shape=(16, 112, 112, 3),
        filters=9,
        kernel_size=(3, 3),
        dense_units=[128, 16, 1],
        use_batch_norm=True,
    )

    # Check input and output shapes
    assert model.input_shape == (None, 16, 112, 112, 3)
    assert model.output_shape == (None, 1)

    # Verify that there is EXACTLY 1 ConvLSTM2D layer with 9 filters
    convlstm_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.ConvLSTM2D)
    ]
    assert len(convlstm_layers) == 1
    assert convlstm_layers[0].filters == 9

    # Verify GlobalAveragePooling2D is present
    gap_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    ]
    assert len(gap_layers) == 1

    # Verify dense layers sequence: 128 -> 16 -> 1
    dense_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)
    ]
    assert len(dense_layers) == 3
    assert dense_layers[0].units == 128
    assert dense_layers[1].units == 16
    assert dense_layers[2].units == 1


def test_dummy_forward_pass_no_training():
    # Run a single forward pass with synthetic dummy batch (no training performed)
    model = build_convlstm_model(
        input_shape=(4, 32, 32, 3),
        filters=9,
        kernel_size=(3, 3),
        dense_units=[128, 16, 1],
    )

    dummy_input = np.random.uniform(-1.0, 1.0, size=(2, 4, 32, 32, 3)).astype(np.float32)
    predictions = model(dummy_input, training=False)

    assert predictions.shape == (2, 1)
    # Output is sigmoid prob
    assert np.all(predictions.numpy() >= 0.0) and np.all(predictions.numpy() <= 1.0)
