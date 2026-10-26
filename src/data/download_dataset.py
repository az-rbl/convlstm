"""Script to download the RWF-2000 dataset from Hugging Face Hub."""

import argparse
import os
import tarfile
from huggingface_hub import snapshot_download


def download_rwf2000(dest_dir: str = "data/raw/RWF-2000") -> str:
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

    # Check if files are in tar archives that need extraction
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.endswith(".tar.gz") or f.endswith(".tar") or f.endswith(".tgz"):
                tar_path = os.path.join(root, f)
                print(f"Extracting {tar_path} while preserving folders...")
                
                archive_base = f.split('.')[0]
                
                with tarfile.open(tar_path, "r:*") as tar_ref:
                    counter = 1
                    for member in tar_ref.getmembers():
                        if member.isfile():
                            # Extract the original folder structure (e.g., "train/class_A")
                            dir_name = os.path.dirname(member.name)
                            
                            # Keep the original file extension
                            ext = os.path.splitext(member.name)[1]
                            
                            # Create the new safe filename
                            new_filename = f"{archive_base}_{counter:05d}{ext}"
                            
                            # Reattach the folder structure to the new filename
                            # If dir_name is empty, this just uses new_filename
                            member.name = os.path.join(dir_name, new_filename) if dir_name else new_filename
                            
                            # Extract using the preserved path and new name
                            tar_ref.extract(member, path=root)
                            counter += 1

    print(f"Dataset successfully prepared at: {dest_dir}")
    return dest_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RWF-2000 dataset from Hugging Face.")
    parser.add_argument("--dest", type=str, default="data/raw/RWF-2000", help="Destination directory")
    args = parser.parse_args()
    download_rwf2000(args.dest)