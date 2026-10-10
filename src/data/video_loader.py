"""Video decoding, uniform temporal sampling, and change detection extraction."""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.data.change_detection import compute_frame_differences_numpy


def load_video_frames(
    video_path: str,
    target_size: Tuple[int, int] = (112, 112),
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Reads raw video frames from file using OpenCV.

    Args:
        video_path: Absolute or relative path to the video file.
        target_size: (height, width) tuple to resize each frame.
        max_frames: Maximum number of frames to read.

    Returns:
        frames: Array of shape (num_frames, H, W, 3) in RGB format, uint8.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")

    frames: List[np.ndarray] = []
    h, w = target_size

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame_rgb.shape[0], frame_rgb.shape[1]) != (h, w):
                frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")

    return np.array(frames, dtype=np.uint8)


def sample_frames_uniformly(
    frames: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Uniformly samples `num_samples` frames across the video length.

    If video has fewer frames than requested, repeats the last frame.
    """
    total_frames = len(frames)
    if total_frames >= num_samples:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        return frames[indices]
    else:
        # Pad with last frame if video is shorter
        pad_count = num_samples - total_frames
        padding = np.repeat(frames[-1:], pad_count, axis=0)
        return np.concatenate([frames, padding], axis=0)


def extract_video_differences(
    video_path: str,
    sequence_length: int = 16,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Loads video, samples sequence_length + 1 frames, and computes sequence_length differences.

    Args:
        video_path: Path to video.
        sequence_length: Number of difference frames T desired.
        target_size: (H, W).

    Returns:
        diffs: Array of shape (sequence_length, H, W, 3) with float32 values in [-1.0, 1.0].
    """
    # Sample sequence_length + 1 frames to compute sequence_length consecutive differences
    raw_frames = load_video_frames(video_path, target_size=target_size)
    sampled = sample_frames_uniformly(raw_frames, num_samples=sequence_length + 1)
    diffs = compute_frame_differences_numpy(sampled, normalize=True)
    return diffs
