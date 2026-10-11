"""Unit tests for Two-Stream converging architecture, gamma correction, and skeleton rendering."""

import numpy as np
import pytest
import tensorflow as tf

from src.data.gamma import apply_gamma_correction, get_gamma_table
from src.data.pose_extractor import SkeletonRenderer
from src.models.two_stream import build_two_stream_model


def test_gamma_correction_values():
    # Test lookup table creation
    table = get_gamma_table(gamma=0.67)
    assert len(table) == 256
    assert table[0] == 0
    assert table[255] == 255
    # gamma < 1 increases brightness of midtones (e.g., 100 -> ~136)
    assert table[100] > 100

    # Test applying to 4D array (T, H, W, C)
    frames = np.full((5, 32, 32, 3), 100, dtype=np.uint8)
    enhanced = apply_gamma_correction(frames, gamma=0.67)
    assert enhanced.shape == frames.shape
    assert enhanced.dtype == np.uint8
    assert np.all(enhanced > 100)


def test_skeleton_renderer_black_background():
    renderer = SkeletonRenderer()
    dummy_frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    rendered = renderer.render_on_black(dummy_frame)

    assert rendered.shape == (64, 64, 3)
    assert rendered.dtype == np.uint8
    # Majority of background pixels should be black (0)
    assert np.mean(rendered == 0) > 0.8
    renderer.close()


def test_two_stream_architecture_layers():
    model = build_two_stream_model(
        input_shape=(5, 32, 32, 3),
        conv_filters=9,
        convlstm_diff_filters=9,
        aggregator_filters=32,
        depth_multiplier=2,
        dense_units=[128, 16, 1],
    )

    # Check dual inputs
    assert len(model.inputs) == 2
    assert tuple(model.inputs[0].shape) == (None, 5, 32, 32, 3)
    assert tuple(model.inputs[1].shape) == (None, 5, 32, 32, 3)
    assert tuple(model.output_shape) == (None, 1)

    layer_names = [l.name for l in model.layers]
    # Check presence of key architectural components from the paper
    assert any("frame_difference" in name for name in layer_names)
    assert any("convlstm_diff" in name for name in layer_names)
    assert any("td_conv2d_pose" in name or "conv2d_pose" in name for name in layer_names)
    assert any("bn_pose_centered" in name for name in layer_names)
    assert any("addition_fusion" in name for name in layer_names)
    assert any("convlstm_aggregator" in name for name in layer_names)
    assert any("depthwise_conv" in name for name in layer_names)
    assert any("global_avg_pool" in name for name in layer_names)


def test_dummy_forward_pass_no_training():
    # Verify forward pass execution with synthetic batches (strictly NO model training)
    model = build_two_stream_model(
        input_shape=(5, 32, 32, 3),
        conv_filters=9,
        convlstm_diff_filters=9,
        aggregator_filters=32,
        depth_multiplier=2,
        dense_units=[128, 16, 1],
    )

    batch_size = 2
    dummy_raw = np.random.uniform(0.0, 1.0, size=(batch_size, 5, 32, 32, 3)).astype(np.float32)
    dummy_skel = np.random.uniform(0.0, 1.0, size=(batch_size, 5, 32, 32, 3)).astype(np.float32)

    predictions = model([dummy_raw, dummy_skel], training=False)

    assert predictions.shape == (batch_size, 1)
    preds_np = predictions.numpy()
    assert np.all(preds_np >= 0.0) and np.all(preds_np <= 1.0)
