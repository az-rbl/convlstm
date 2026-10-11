"""Dataset generators and batch loaders for RWF-2000 video classification (supports two-stream pairs)."""

import glob
import os
import random
from typing import Dict, List, Optional, Tuple, Union
import keras
import numpy as np
import tensorflow as tf

from src.data.video_loader import extract_video_differences, load_video_frames, sample_frames_uniformly


class RWF2000Sequence(keras.utils.Sequence):
    """Keras Sequence for lazy loading and multi-stream batch extraction from RWF-2000 videos."""

    def __init__(
        self,
        split_dir: str,
        sequence_length: int = 16,
        target_size: Tuple[int, int] = (112, 112),
        batch_size: int = 8,
        shuffle: bool = True,
        class_mapping: Optional[Dict[str, int]] = None,
        augment: bool = False,
        two_stream: bool = True,
    ):
        """Initializes the dataset sequence.

        Args:
            split_dir: Directory pointing to train or val split (e.g. 'RWF-2000/train').
            sequence_length: Number of difference frames T.
            target_size: (H, W).
            batch_size: Batch size.
            shuffle: Whether to shuffle sample order at each epoch end.
            class_mapping: Map from folder name to integer label (default: NonFight=0, Fight=1).
            augment: Whether to apply random spatial flipping.
            two_stream: If True, returns ([raw_video, skeleton_video], labels).
                        If False, returns (diff_video, labels).
        """
        super().__init__()
        self.split_dir = split_dir
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.two_stream = two_stream
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

            # Exclude preprocessed skeleton videos from sample list
            video_paths = [
                p for p in video_paths
                if not p.endswith("_skeletons.avi") and not p.endswith("_skeletons.mp4")
            ]

            for path in sorted(video_paths):
                samples.append((path, label))

        return samples

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / float(self.batch_size)))

    def _load_sample(self, video_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Loads raw frames and paired skeleton frames."""
        # Raw frames (sample sequence_length + 1 frames)
        raw_frames = load_video_frames(video_path, target_size=self.target_size)
        raw_sampled = sample_frames_uniformly(raw_frames, num_samples=self.sequence_length + 1)
        raw_tensor = (raw_sampled.astype(np.float32) / 255.0)

        skeleton_tensor = None
        if self.two_stream:
            # Check for paired skeleton video
            base, ext = os.path.splitext(video_path)
            skeleton_path = f"{base}_skeletons{ext}"
            if os.path.exists(skeleton_path):
                skel_frames = load_video_frames(skeleton_path, target_size=self.target_size)
                skel_sampled = sample_frames_uniformly(skel_frames, num_samples=self.sequence_length + 1)
                skeleton_tensor = (skel_sampled.astype(np.float32) / 255.0)
            else:
                # Black background canvas fallback
                skeleton_tensor = np.zeros_like(raw_tensor)

        return raw_tensor, skeleton_tensor

    def __getitem__(self, idx: int) -> Union[Tuple[List[np.ndarray], np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_raw = []
        batch_skel = []
        batch_diff = []
        batch_y = []

        for b_idx in batch_indices:
            video_path, label = self.samples[b_idx]
            try:
                if self.two_stream:
                    raw_tensor, skel_tensor = self._load_sample(video_path)
                    do_flip = self.augment and random.random() > 0.5
                    if do_flip:
                        raw_tensor = np.flip(raw_tensor, axis=2)
                        if skel_tensor is not None:
                            skel_tensor = np.flip(skel_tensor, axis=2)
                    batch_raw.append(raw_tensor)
                    batch_skel.append(skel_tensor)
                else:
                    diffs = extract_video_differences(
                        video_path=video_path,
                        sequence_length=self.sequence_length,
                        target_size=self.target_size,
                    )
                    if self.augment and random.random() > 0.5:
                        diffs = np.flip(diffs, axis=2)
                    batch_diff.append(diffs)

                batch_y.append(label)
            except Exception as e:
                # Fallback on zeros for corrupt video
                h, w = self.target_size
                if self.two_stream:
                    zeros = np.zeros((self.sequence_length + 1, h, w, 3), dtype=np.float32)
                    batch_raw.append(zeros)
                    batch_skel.append(zeros)
                else:
                    zeros = np.zeros((self.sequence_length, h, w, 3), dtype=np.float32)
                    batch_diff.append(zeros)
                batch_y.append(label)

        y = np.array(batch_y, dtype=np.float32)

        if self.two_stream:
            x_raw = np.array(batch_raw, dtype=np.float32)
            x_skel = np.array(batch_skel, dtype=np.float32)
            return [x_raw, x_skel], y
        else:
            x_diff = np.array(batch_diff, dtype=np.float32)
            return x_diff, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_dataset(
    sequence: RWF2000Sequence,
) -> tf.data.Dataset:
    """Wraps RWF2000Sequence into a tf.data.Dataset."""
    h, w = sequence.target_size
    seq_len = sequence.sequence_length

    if sequence.two_stream:
        def generator():
            for i in range(len(sequence)):
                (x_raw, x_skel), y = sequence[i]
                for raw, skel, label in zip(x_raw, x_skel, y):
                    yield (raw, skel), label

        output_signature = (
            (
                tf.TensorSpec(shape=(seq_len + 1, h, w, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len + 1, h, w, 3), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    else:
        def generator():
            for i in range(len(sequence)):
                x_diff, y = sequence[i]
                for diff, label in zip(x_diff, y):
                    yield diff, label

        output_signature = (
            tf.TensorSpec(shape=(seq_len, h, w, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(sequence.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
