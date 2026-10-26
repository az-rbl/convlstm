"""Evaluation script for trained ConvLSTM video classification models."""

import argparse
import os
import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import AppConfig
from src.data.generator import RWF2000Sequence


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM model on validation set")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_convlstm_model.keras",
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def evaluate():
    args = parse_args()
    config = AppConfig.from_yaml(args.config)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    val_dir = os.path.join(config.data.dataset_path, config.data.val_split)
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found at: {val_dir}")

    skeletons_path = getattr(config.data, "skeletons_path", None)
    skeletons_val_dir = os.path.join(skeletons_path, config.data.val_split) if skeletons_path else None

    print(f"Loading model from: {args.model_path}")
    model = keras.models.load_model(args.model_path)

    val_seq = RWF2000Sequence(
        split_dir=val_dir,
        skeletons_dir=skeletons_val_dir,
        sequence_length=config.data.sequence_length,
        target_size=(config.data.image_size[0], config.data.image_size[1]),
        batch_size=config.data.batch_size,
        shuffle=False,
        augment=False,
    )

    print(f"Evaluating on {len(val_seq.samples)} validation videos...")
    y_true = []
    y_pred_probs = []

    for i in range(len(val_seq)):
        batch_x, batch_y = val_seq[i]
        preds = model.predict(batch_x, verbose=0).flatten()
        y_pred_probs.extend(preds)
        y_true.extend(batch_y)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)

    acc = np.mean(y_true == y_pred_binary)
    auc = roc_auc_score(y_true, y_pred_probs)
    cm = confusion_matrix(y_true, y_pred_binary)

    print("\n" + "=" * 50)
    print("              EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["NonFight", "Fight"]))
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
