"""Unit tests strictly verifying replication alignment with Garcia-Cobo & SanMiguel (CVIU 2023)."""

import os
import subprocess
import sys
import numpy as np
import pytest
import tensorflow as tf

from src.config import AppConfig
from src.models.two_stream import build_two_stream_from_config, build_two_stream_model


def test_garcia_exact_parameter_count():
    """Verify exact 62,583 trainable parameters matching Table 7 in Garcia-Cobo & SanMiguel (CVIU 2023)."""
    model = build_two_stream_model(
        input_shape=(51, 100, 100, 3),
        conv_filters=9,
        convlstm_diff_filters=9,
        aggregator_filters=32,
        depth_multiplier=2,
        padding="valid",
        dense_units=[128, 16, 1],
        dropout_rate=0.0,
    )

    trainable_params = sum(w.numpy().size for w in model.trainable_weights)
    non_trainable_params = sum(w.numpy().size for w in model.non_trainable_weights)

    assert trainable_params == 62583, (
        f"Trainable parameters ({trainable_params}) must match exact 62,583 from paper Table 7"
    )
    # Centered batch norm on pose stream: 18 non-trainable; diff batch norm: 6 trainable, 6 non-trainable -> total 24
    assert non_trainable_params == 24


def test_garcia_intermediate_tensor_shapes():
    """Verify intermediate tensor dimensions match the valid-padding geometric shrinkage."""
    model = build_two_stream_model(
        input_shape=(51, 100, 100, 3),
        conv_filters=9,
        convlstm_diff_filters=9,
        aggregator_filters=32,
        depth_multiplier=2,
        padding="valid",
        dense_units=[128, 16, 1],
        dropout_rate=0.0,
    )

    layer_map = {layer.name: layer for layer in model.layers}

    # 1. Temporal differences and skeleton slice
    assert tuple(layer_map["frame_difference_lambda"].output.shape) == (None, 50, 100, 100, 3)
    assert tuple(layer_map["skeleton_slice_lambda"].output.shape) == (None, 50, 100, 100, 3)

    # 2. Streams before fusion (spatial size 100 -> 98 due to 3x3 valid conv)
    assert tuple(layer_map["convlstm_diff_stream"].output.shape) == (None, 50, 98, 98, 9)
    assert tuple(layer_map["td_conv2d_pose"].output.shape) == (None, 50, 98, 98, 9)
    assert tuple(layer_map["addition_fusion"].output.shape) == (None, 50, 98, 98, 9)

    # 3. Spatiotemporal Aggregator (spatial size 98 -> 96, time collapsed)
    assert tuple(layer_map["convlstm_aggregator"].output.shape) == (None, 96, 96, 32)

    # 4. Depthwise post-processing (spatial size 96 -> 94, channels 32 * 2 = 64)
    assert tuple(layer_map["depthwise_conv_post"].output.shape) == (None, 94, 94, 64)

    # 5. Global pooling and dense classification sequence
    assert tuple(layer_map["global_avg_pool"].output.shape) == (None, 64)
    assert tuple(layer_map["dense_128"].output.shape) == (None, 128)
    assert tuple(layer_map["dense_16"].output.shape) == (None, 16)
    assert tuple(layer_map["output_dense_1"].output.shape) == (None, 1)


def test_garcia_dummy_forward_pass():
    """Verify synthetic forward pass runs cleanly with valid output probabilities."""
    model = build_two_stream_model(
        input_shape=(51, 100, 100, 3),
        padding="valid",
        dropout_rate=0.0,
    )

    batch_size = 2
    dummy_raw = np.random.uniform(0.0, 255.0, size=(batch_size, 51, 100, 100, 3)).astype(np.float32)
    dummy_skel = np.random.uniform(0.0, 255.0, size=(batch_size, 51, 100, 100, 3)).astype(np.float32)

    out = model([dummy_raw, dummy_skel], training=False)
    assert out.shape == (batch_size, 1)

    out_np = out.numpy()
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)


def test_garcia_config_yaml_loading():
    """Verify config/garcia_cviu_2023.yaml loads and builds model with 62,583 params."""
    config_path = "config/garcia_cviu_2023.yaml"
    assert os.path.exists(config_path)

    config = AppConfig.from_yaml(config_path)
    assert config.data.sequence_length == 50
    assert config.data.image_size == [100, 100]
    assert config.data.batch_size == 10
    assert config.data.normalize is False
    assert config.model.type == "two_stream"
    assert config.model.padding == "valid"
    assert config.model.dropout_rate == 0.0
    assert config.training.learning_rate == 0.001
    assert config.training.epochs == 50
    assert config.training.early_stopping.enabled is False
    assert config.training.reduce_lr_on_plateau.enabled is False

    model = build_two_stream_from_config(config)
    trainable_params = sum(w.numpy().size for w in model.trainable_weights)
    assert trainable_params == 62583


def test_train_original_dry_run_cli():
    """Verify src/train_original.py --dry_run CLI command completes cleanly."""
    cmd = [sys.executable, "-m", "src.train_original", "--dry_run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"train_original failed: {result.stderr}"
    assert "Total Trainable Parameters: 62,583" in result.stdout
    assert "Model and pipeline verified successfully" in result.stdout
