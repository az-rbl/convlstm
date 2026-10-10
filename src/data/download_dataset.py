"""Script to download the RWF-2000 dataset from Hugging Face Hub."""

import argparse
import os
import zipfile
from huggingface_hub import snapshot_download


def download_rwf2000(dest_dir: str = "RWF-2000") -> str:
    """Downloads RWF-2000 from Hugging Face (DanJoshua/RWF-2000) into dest_dir.

    Args:
        dest_dir: Destination folder.

    Returns:
        Path to downloaded folder.
    """
    print(f"Downloading RWF-2000 from Hugging Face: DanJoshua/RWF-2000 into {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)

    local_dir = snapshot_download(
        repo_id="DanJoshua/RWF-2000",
        repo_type="dataset",
        local_dir=dest_dir,
    )

    # Check if files are in zip archives that need extraction
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(root)

    print(f"Dataset successfully prepared at: {dest_dir}")
    return dest_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RWF-2000 dataset from Hugging Face.")
    parser.add_argument("--dest", type=str, default="RWF-2000", help="Destination directory")
    args = parser.parse_args()
    download_rwf2000(args.dest)
