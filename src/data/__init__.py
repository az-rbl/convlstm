"""Data preprocessing, change detection, and video loaders."""

from src.data.change_detection import (
    compute_frame_differences_numpy,
    compute_frame_differences_tf,
)
from src.data.video_loader import extract_video_differences, load_video_frames
from src.data.generator import RWF2000Sequence, create_tf_dataset

# Convenient alias
compute_frame_differences = compute_frame_differences_numpy

__all__ = [
    "compute_frame_differences",
    "compute_frame_differences_numpy",
    "compute_frame_differences_tf",
    "load_video_frames",
    "extract_video_differences",
    "RWF2000Sequence",
    "create_tf_dataset",
]
