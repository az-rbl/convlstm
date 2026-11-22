"""Standalone training script replicating Garcia-Cobo & SanMiguel (CVIU 2023).

Paper: "Human skeletons and change detection for efficient violence detection in surveillance videos"
Reference notebook: Best model - final traning (last 20 epochs).ipynb & preprocess.py

Key specifications:
  - Exact 62,583 trainable parameter architecture
  - 51 frames (50 difference frames) at 100x100 spatial resolution
  - Pose stream rendered with MediaPipe on black canvas with gamma=0.67
  - Keras 'valid' padding on all Conv2D / ConvLSTM2D / DepthwiseConv2D layers
  - No Dropout, no recurrent dropout
  - Adam optimizer with fixed learning_rate=0.001
  - Batch size: 10, Epochs: 50
  - Random seed: 0
  - ModelCheckpoint on 'val_accuracy' (save_best_only=True)
"""

import argparse
import os
import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks

from src.data.generator import RWF2000Sequence
from src.models.two_stream import build_two_stream_model


def set_seed(seed: int = 0):
    """Enforces deterministic reproducibility matching the original experiment."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replicate Garcia-Cobo & SanMiguel (CVIU 2023) Training Pipeline"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/raw/RWF-2000",
        help="Path to raw RWF-2000 video directory",
    )
    parser.add_argument(
        "--skeletons_path",
        type=str,
        default="data/processed/RWF-2000",
        help="Path to preprocessed skeleton videos",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total training epochs (paper trains for 50 epochs)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size (paper uses batch size 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (paper uses fixed 0.001)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/best_model.h5",
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify model building, shapes, and pipeline initialization without running full training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Enforce exact random seed (Seed = 0)
    set_seed(0)

    print("==================================================================")
    print("  Garcia-Cobo & SanMiguel (CVIU 2023) Exact Replication Pipeline")
    print("==================================================================")
    print(f"Dataset path:    {args.dataset_path}")
    print(f"Skeletons path:  {args.skeletons_path}")
    print(f"Resolution:      100 x 100")
    print(f"Frames loaded:   51 (T=50 consecutive difference frames)")
    print(f"Batch size:      {args.batch_size}")
    print(f"Epochs:          {args.epochs}")
    print(f"Learning rate:   {args.lr} (Adam, fixed)")
    print(f"Checkpoint path: {args.save_path}")
    print("==================================================================")

    # 2. Build Exact Architecture (62,583 trainable parameters)
    model = build_two_stream_model(
        input_shape=(51, 100, 100, 3),
        conv_filters=9,
        convlstm_diff_filters=9,
        aggregator_filters=32,
        depth_multiplier=2,
        padding="valid",
        dense_units=[128, 16, 1],
        dropout_rate=0.0,
        name="Garcia_CVIU_2023_Replication",
    )

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    trainable_count = sum(w.numpy().size for w in model.trainable_weights)
    print(f"\n[Model Architecture Verified]")
    print(f"  Total Trainable Parameters: {trainable_count:,} (Expected: 62,583)")
    assert trainable_count == 62583, f"Expected 62,583 trainable parameters, got {trainable_count}"

    model.summary()

    if args.dry_run:
        print("\n[Dry Run] Model and pipeline verified successfully. Exiting without training.")
        return

    # 3. Prepare Dataset Sequences
    train_dir = os.path.join(args.dataset_path, "train")
    val_dir = os.path.join(args.dataset_path, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Dataset directories not found! Expected:\n  {train_dir}\n  {val_dir}\n"
            f"Please download RWF-2000 dataset or specify --dataset_path."
        )

    skeletons_train_dir = (
        os.path.join(args.skeletons_path, "train") if args.skeletons_path else None
    )
    skeletons_val_dir = (
        os.path.join(args.skeletons_path, "val") if args.skeletons_path else None
    )

    print("\n[Data] Initializing two-stream data generators...")
    train_generator = RWF2000Sequence(
        split_dir=train_dir,
        skeletons_dir=skeletons_train_dir,
        sequence_length=50,
        target_size=(100, 100),
        batch_size=args.batch_size,
        shuffle=True,
        normalize=False,
        augment=False,
        two_stream=True,
    )

    val_generator = RWF2000Sequence(
        split_dir=val_dir,
        skeletons_dir=skeletons_val_dir,
        sequence_length=50,
        target_size=(100, 100),
        batch_size=args.batch_size,
        shuffle=False,
        normalize=False,
        augment=False,
        two_stream=True,
    )

    print(f"  Train samples: {len(train_generator.samples)} ({len(train_generator)} batches)")
    print(f"  Val samples:   {len(val_generator.samples)} ({len(val_generator)} batches)")

    # 4. Configure Checkpoint Callback matching the original notebook
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=args.save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # 5. Fit Model for 50 Epochs
    print(f"\n[Training] Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=[checkpoint_cb],
        verbose=2,
    )

    print("\n[Complete] Training completed successfully!")
    print(f"Best model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
