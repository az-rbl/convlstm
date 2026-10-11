"""Gamma correction module for video enhancement."""

import cv2
import numpy as np


def get_gamma_table(gamma: float = 0.67) -> np.ndarray:
    """Computes a 256-element lookup table for gamma correction.

    Args:
        gamma: Gamma correction factor (default: 0.67 as proposed in paper).

    Returns:
        Look-up table array of dtype uint8.
    """
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}")
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table


# Precomputed lookup table for default gamma = 0.67
DEFAULT_GAMMA_TABLE = get_gamma_table(0.67)


def apply_gamma_correction(
    frames: np.ndarray,
    gamma: float = 0.67,
) -> np.ndarray:
    """Applies gamma correction to video frames using OpenCV LUT for maximum speed.

    Args:
        frames: Array of shape (T, H, W, 3) or (H, W, 3), uint8.
        gamma: Gamma value (default: 0.67).

    Returns:
        Enhanced frames array with the same shape and uint8 dtype.
    """
    if gamma == 1.0:
        return frames

    table = DEFAULT_GAMMA_TABLE if gamma == 0.67 else get_gamma_table(gamma)

    original_shape = frames.shape
    if frames.ndim == 4:
        # Video sequence (T, H, W, C)
        out = np.empty_like(frames)
        for t in range(frames.shape[0]):
            out[t] = cv2.LUT(frames[t], table)
        return out
    elif frames.ndim == 3:
        # Single frame (H, W, C)
        return cv2.LUT(frames, table)
    else:
        raise ValueError(f"Expected 3D or 4D array, got ndim={frames.ndim}")
