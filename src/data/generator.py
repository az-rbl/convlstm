"""Dataset generators and batch loaders for RWF-2000 video classification."""

import glob
import os
import random
from typing import Dict, List, Optional, Tuple
import keras
import numpy as np
import tensorflow as tf

from src.data.video_loader import extract_video_differences


class RWF2000Sequence(keras.utils.Sequence):
    """Keras Sequence for lazy loading and change detection extraction from RWF-2000 videos."""

    def __init__(
        self,
        split_dir: str,
        sequence_length: int = 16,
        target_size: Tuple[int, int] = (112, 112),
        batch_size: int = 8,
        shuffle: bool = True,
        class_mapping: Optional[Dict[str, int]] = None,
        augment: bool = False,
    ):
        """Initializes the dataset sequence.

        Args:
            split_dir: Directory pointing to train or val split (e.g. 'RWF-2000/train').
            sequence_length: Number of difference frames T.
            target_size: (H, W).
            batch_size: Batch size.
            shuffle: Whether to shuffle sample order at each epoch end.
            class_mapping: Map from folder name to integer label (default: NonFight=0, Fight=1).
            augment: Whether to apply random spatial flipping to difference frames.
        """
        super().__init__()
        self.split_dir = split_dir
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.class_mapping = class_mapping or {"NonFight": 0, "Fight": 1}

        self.samples: List[Tuple[str, int]] = self._collect_samples()
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _collect_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        valid_extensions = ("*.avi", "*.mp4", "*.mkv", "*.mov")

        for class_name, label in self.class_mapping.items():
            class_folder = os.path.join(self.split_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            video_paths: List[str] = []
            for ext in valid_extensions:
                video_paths.extend(glob.glob(os.path.join(class_folder, ext)))
                video_paths.extend(glob.glob(os.path.join(class_folder, ext.upper())))

            for path in sorted(video_paths):
                samples.append((path, label))

        return samples

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = []
        batch_y = []

        for b_idx in batch_indices:
            video_path, label = self.samples[b_idx]
            try:
                diffs = extract_video_differences(
                    video_path=video_path,
                    sequence_length=self.sequence_length,
                    target_size=self.target_size,
                )
                if self.augment and random.random() > 0.5:
                    # Random horizontal flip across all frames consistently
                    diffs = np.flip(diffs, axis=2)
                batch_x.append(diffs)
                batch_y.append(label)
            except Exception as e:
                # If a corrupted video occurs, fallback to zeros to preserve batch shape
                zero_diffs = np.zeros(
                    (self.sequence_length, self.target_size[0], self.target_size[1], 3),
                    dtype=np.float32,
                )
                batch_x.append(zero_diffs)
                batch_y.append(label)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_dataset(
    sequence: RWF2000Sequence,
) -> tf.data.Dataset:
    """Wraps RWF2000Sequence into a high-performance tf.data.Dataset."""
    h, w = sequence.target_size
    seq_len = sequence.sequence_length

    def generator():
        for i in range(len(sequence)):
            x, y = sequence[i]
            for bx, by in zip(x, y):
                yield bx, by

    output_signature = (
        tf.TensorSpec(shape=(seq_len, h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(sequence.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
