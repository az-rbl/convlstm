"""Single video inference script using two-stream ConvLSTM with pose & change detection."""

import argparse
import os
import keras
import numpy as np

from src.config import AppConfig
from src.data.pose_extractor import process_video_skeletons
from src.data.video_loader import extract_video_differences, load_video_frames, sample_frames_uniformly


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

    target_size = (config.data.image_size[0], config.data.image_size[1])
    is_two_stream = isinstance(model.input_shape, list) and len(model.input_shape) == 2

    if is_two_stream:
        print(f"[Inference] Two-stream model detected. Loading raw video and extracting pose...")
        raw_frames = load_video_frames(args.video, target_size=target_size)
        raw_sampled = sample_frames_uniformly(raw_frames, num_samples=config.data.sequence_length + 1)
        raw_tensor = np.expand_dims(raw_sampled.astype(np.float32) / 255.0, axis=0)

        # Check for pre-existing skeleton video or extract on the fly
        base, ext = os.path.splitext(args.video)
        skeleton_path = f"{base}_skeletons{ext}"
        if os.path.exists(skeleton_path):
            print(f"[Inference] Found pre-extracted skeletons at: {skeleton_path}")
            skel_frames = load_video_frames(skeleton_path, target_size=target_size)
            skel_sampled = sample_frames_uniformly(skel_frames, num_samples=config.data.sequence_length + 1)
        else:
            print("[Inference] Extracting skeletons with gamma correction (0.67)...")
            skel_frames = process_video_skeletons(
                video_path=args.video,
                gamma=getattr(config.model, "gamma", 0.67),
                target_size=target_size,
            )
            skel_sampled = sample_frames_uniformly(skel_frames, num_samples=config.data.sequence_length + 1)

        skel_tensor = np.expand_dims(skel_sampled.astype(np.float32) / 255.0, axis=0)
        model_inputs = [raw_tensor, skel_tensor]
    else:
        print(f"[Inference] Single-stream model detected. Extracting frame differences...")
        diffs = extract_video_differences(
            video_path=args.video,
            sequence_length=config.data.sequence_length,
            target_size=target_size,
        )
        model_inputs = np.expand_dims(diffs, axis=0)

    prob = float(model.predict(model_inputs, verbose=0)[0][0])
    label = "Fight" if prob >= 0.5 else "NonFight"

    print("\n" + "=" * 40)
    print(f"Video:      {args.video}")
    print(f"Prediction: {label}")
    print(f"Confidence: {prob * 100:.2f}% (Fight probability)")
    print("=" * 40)


if __name__ == "__main__":
    predict()
