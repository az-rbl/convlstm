"""Single video inference script using change detection and trained ConvLSTM."""

import argparse
import os
import keras
import numpy as np

from src.config import AppConfig
from src.data.video_loader import extract_video_differences


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
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


def predict():
    args = parse_args()
    config = AppConfig.from_yaml(args.config)

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print(f"Loading model: {args.model_path}")
    model = keras.models.load_model(args.model_path)

    print(f"Extracting consecutive frame differences from: {args.video}")
    diffs = extract_video_differences(
        video_path=args.video,
        sequence_length=config.data.sequence_length,
        target_size=(config.data.image_size[0], config.data.image_size[1]),
    )

    # Expand batch dimension: (1, T, H, W, 3)
    input_tensor = np.expand_dims(diffs, axis=0)

    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    label = "Fight" if prob >= 0.5 else "NonFight"

    print("\n" + "=" * 40)
    print(f"Video:      {args.video}")
    print(f"Prediction: {label}")
    print(f"Confidence: {prob * 100:.2f}% (Fight probability)")
    print("=" * 40)


if __name__ == "__main__":
    predict()
