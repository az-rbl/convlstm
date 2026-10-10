"""Unit tests for Change Detection pipeline and Deep ConvLSTM Model."""

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


def test_convlstm_depth_9_model_architecture():
    # Build a depth-9 model with small spatial shape for rapid test
    model = build_convlstm_model(
        input_shape=(4, 32, 32, 3),
        depth=9,
        filters=[8] * 9,
        dense_units=[128, 16, 1],
        use_batch_norm=True,
    )

    # Check input and output shapes
    assert model.input_shape == (None, 4, 32, 32, 3)
    assert model.output_shape == (None, 1)

    # Verify that there are exactly 9 ConvLSTM2D layers
    convlstm_layers = [
        layer for layer in model.layers if "conv_lstm" in layer.name or "convlstm" in layer.name
    ]
    assert len(convlstm_layers) == 9

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
        depth=9,
        filters=[8] * 9,
        dense_units=[128, 16, 1],
    )

    dummy_input = np.random.uniform(-1.0, 1.0, size=(2, 4, 32, 32, 3)).astype(np.float32)
    predictions = model(dummy_input, training=False)

    assert predictions.shape == (2, 1)
    # Output is sigmoid prob
    assert np.all(predictions.numpy() >= 0.0) and np.all(predictions.numpy() <= 1.0)
