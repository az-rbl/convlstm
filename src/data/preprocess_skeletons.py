"""Batch offline preprocessing utility to render skeletons into a dedicated, tidy directory structure.

Preserves dataset cleanliness by mirroring the input folder hierarchy into a separate
output directory (e.g. RWF-2000/ -> RWF-2000_skeletons/), keeping raw video and preprocessed
skeleton videos completely separate.
"""

import argparse
import glob
import os
from typing import Tuple
from tqdm import tqdm

from src.data.pose_extractor import SkeletonRenderer, process_video_skeletons


def preprocess_dataset(
    dataset_dir: str = "data/raw/RWF-2000",
    output_dir: str = "data/processed/RWF-2000",
    gamma: float = 0.67,
    image_size: int = 112,
    num_poses: int = 4,
):
    """Discovers all videos in dataset_dir and renders corresponding multi-person skeletons

    into mirrored directories inside output_dir.

    Args:
        dataset_dir: Root raw video directory (e.g. 'RWF-2000').
        output_dir: Dedicated directory for preprocessed skeleton videos (e.g. 'RWF-2000_skeletons').
        gamma: Gamma correction factor (default: 0.67).
        image_size: Spatial resolution (H=W).
        num_poses: Max number of people detected per frame.
    """
    valid_extensions = ("*.avi", "*.mp4", "*.mkv", "*.mov")
    all_videos = []
    for ext in valid_extensions:
        all_videos.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
        all_videos.extend(glob.glob(os.path.join(dataset_dir, "**", ext.upper()), recursive=True))

    all_videos = sorted(list(set(all_videos)))
    if not all_videos:
        print(f"[!] No video files found in: {dataset_dir}")
        return

    print("==================================================")
    print("  Tidy Skeleton Preprocessing Pipeline")
    print("==================================================")
    print(f"Source Directory: {dataset_dir}")
    print(f"Target Directory: {output_dir} (Separated & Clean)")
    print(f"Videos Found:     {len(all_videos)}")
    print(f"Gamma Correction: {gamma}")
    print(f"Target Size:      {image_size}x{image_size}")
    print(f"Max Poses/Frame:  {num_poses}")
    print("==================================================")

    target_size: Tuple[int, int] = (image_size, image_size)
    renderer = SkeletonRenderer(num_poses=num_poses)

    skipped = 0
    processed = 0

    try:
        for video_path in tqdm(all_videos, desc="Rendering skeletons on black background"):
            # Compute mirrored relative path
            rel_path = os.path.relpath(video_path, dataset_dir)
            out_video_path = os.path.join(output_dir, rel_path)

            if os.path.exists(out_video_path):
                skipped += 1
                continue

            try:
                process_video_skeletons(
                    video_path=video_path,
                    output_path=out_video_path,
                    gamma=gamma,
                    target_size=target_size,
                    renderer=renderer,
                )
                processed += 1
            except Exception as e:
                print(f"\n[Error] Processing {video_path}: {e}")
    finally:
        renderer.close()

    print(f"\nPreprocessing finished! Processed: {processed}, Already existed: {skipped}")
    print(f"Clean skeleton dataset stored in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mirror raw dataset into clean, dedicated skeleton video directory."
    )
    parser.add_argument("--dataset_dir", type=str, default="data/raw/RWF-2000", help="Root input video directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/RWF-2000",
        help="Dedicated tidy output directory for skeleton videos",
    )
    parser.add_argument("--gamma", type=float, default=0.67, help="Gamma factor (default: 0.67)")
    parser.add_argument("--size", type=int, default=112, help="Spatial resolution (H=W)")
    parser.add_argument("--num_poses", type=int, default=4, help="Max poses per frame")
    args = parser.parse_args()

    preprocess_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        gamma=args.gamma,
        image_size=args.size,
        num_poses=args.num_poses,
    )
