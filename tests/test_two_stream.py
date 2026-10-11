"""Unit tests for Two-Stream converging architecture, gamma correction, and skeleton rendering."""

import os
import tempfile
import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.data.gamma import apply_gamma_correction, get_gamma_table
from src.data.generator import RWF2000Sequence
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
    dummy_frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    
    # 1. Native rendering
    rendered_native = renderer.render_on_black(dummy_frame)
    assert rendered_native.shape == (240, 320, 3)
    assert rendered_native.dtype == np.uint8

    # 2. Direct rendering to target canvas (112, 112) without image downsampling
    rendered_target = renderer.render_on_black(dummy_frame, target_size=(112, 112))
    assert rendered_target.shape == (112, 112, 3)
    assert rendered_target.dtype == np.uint8
    assert np.mean(rendered_target == 0) > 0.8
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


def test_tidy_skeleton_directory_lookup():
    # Test path resolution for separate, mirrored skeleton directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_train = os.path.join(tmp_dir, "RWF-2000", "train")
        skel_train = os.path.join(tmp_dir, "RWF-2000_skeletons", "train")
        os.makedirs(os.path.join(raw_train, "Fight"), exist_ok=True)
        os.makedirs(os.path.join(skel_train, "Fight"), exist_ok=True)

        raw_vid = os.path.join(raw_train, "Fight", "vid_01.avi")
        skel_vid = os.path.join(skel_train, "Fight", "vid_01.avi")

        # Create dummy video files
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        w_raw = cv2.VideoWriter(raw_vid, fourcc, 10, (32, 32))
        w_skel = cv2.VideoWriter(skel_vid, fourcc, 10, (32, 32))
        for _ in range(5):
            w_raw.write(np.zeros((32, 32, 3), dtype=np.uint8))
            w_skel.write(np.full((32, 32, 3), 255, dtype=np.uint8))
        w_raw.release()
        w_skel.release()

        seq = RWF2000Sequence(
            split_dir=raw_train,
            skeletons_dir=skel_train,
            sequence_length=3,
            target_size=(32, 32),
            batch_size=1,
            two_stream=True,
        )

        found_path = seq._find_skeleton_path(raw_vid)
        assert found_path == skel_vid
        assert os.path.exists(found_path)

        # Verify batch return structure is tuple (not list) to prevent TypeError with tf.data.Dataset
        batch_inputs, batch_y = seq[0]
        assert isinstance(batch_inputs, tuple)
        assert not isinstance(batch_inputs, list)
        assert len(batch_inputs) == 2
        assert batch_inputs[0].shape == (1, 4, 32, 32, 3)
        assert batch_inputs[1].shape == (1, 4, 32, 32, 3)
        assert batch_y.shape == (1,)


def test_nested_skeleton_directory_lookup():
    """Verify lookup succeeds when skeleton directory mirrors the nested RWF-2000 structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_train = os.path.join(tmp_dir, "RWF-2000", "train")
        # Mirrored nested structure: RWF-2000_skeletons/RWF-2000/train
        skel_train = os.path.join(tmp_dir, "RWF-2000_skeletons", "RWF-2000", "train")
        os.makedirs(os.path.join(raw_train, "Fight"), exist_ok=True)
        os.makedirs(os.path.join(skel_train, "Fight"), exist_ok=True)

        raw_vid = os.path.join(raw_train, "Fight", "vid_01.avi")
        skel_vid = os.path.join(skel_train, "Fight", "vid_01.avi")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        w_raw = cv2.VideoWriter(raw_vid, fourcc, 10, (32, 32))
        w_skel = cv2.VideoWriter(skel_vid, fourcc, 10, (32, 32))
        for _ in range(5):
            w_raw.write(np.zeros((32, 32, 3), dtype=np.uint8))
            w_skel.write(np.full((32, 32, 3), 255, dtype=np.uint8))
        w_raw.release()
        w_skel.release()

        # skeletons_dir passed as top-level without subfolder
        seq = RWF2000Sequence(
            split_dir=raw_train,
            skeletons_dir=os.path.join(tmp_dir, "RWF-2000_skeletons"),
            sequence_length=3,
            target_size=(32, 32),
            batch_size=1,
            two_stream=True,
        )

        found_path = seq._find_skeleton_path(raw_vid)
        assert found_path == skel_vid
        assert os.path.exists(found_path)

