"""Batch offline preprocessing utility for generating skeleton videos on black backgrounds.

Applies:
  1. Gamma correction (gamma = 0.67) to enhance contrast.
  2. Pose estimation rendering multi-person skeletons with distinct colored limbs
     over a solid black background.
  3. Writes paired `*_skeletons.avi` alongside original videos.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.data.pose_extractor import process_video_skeletons


def process_single_video(video_path: str, gamma: float = 0.67, target_size=(112, 112)) -> str:
    """Processes a single video into its corresponding _skeletons.avi pair."""
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_skeletons{ext}"

    if os.path.exists(output_path):
        return f"[Skipped] Already exists: {output_path}"

    try:
        process_video_skeletons(
            video_path=video_path,
            output_path=output_path,
            gamma=gamma,
            target_size=target_size,
        )
        return f"[Done] {output_path}"
    except Exception as e:
        return f"[Error] {video_path}: {e}"


def preprocess_dataset(
    dataset_dir: str = "RWF-2000",
    gamma: float = 0.67,
    image_size: int = 112,
    num_workers: int = 2,
):
    """Finds all non-skeleton video files in dataset_dir and renders skeleton pairs."""
    patterns = [
        os.path.join(dataset_dir, "**", "*.avi"),
        os.path.join(dataset_dir, "**", "*.mp4"),
    ]
    all_videos = []
    for pat in patterns:
        all_videos.extend(glob.glob(pat, recursive=True))

    # Filter out already existing skeleton videos
    source_videos = [v for v in all_videos if not v.endswith("_skeletons.avi") and not v.endswith("_skeletons.mp4")]
    source_videos.sort()

    print(f"Found {len(source_videos)} videos to process in: {dataset_dir}")
    print(f"Applying Gamma Correction: gamma={gamma} | Canvas Size: {image_size}x{image_size}")

    target_size = (image_size, image_size)
    for v in tqdm(source_videos, desc="Rendering skeletons on black background"):
        process_single_video(v, gamma=gamma, target_size=target_size)

    print("\nSkeleton preprocessing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess videos into skeleton videos on black background.")
    parser.add_argument("--dataset_dir", type=str, default="RWF-2000", help="Root dataset directory")
    parser.add_argument("--gamma", type=float, default=0.67, help="Gamma factor (default: 0.67)")
    parser.add_argument("--size", type=int, default=112, help="Spatial dimension (H=W)")
    args = parser.parse_args()

    preprocess_dataset(dataset_dir=args.dataset_dir, gamma=args.gamma, image_size=args.size)
