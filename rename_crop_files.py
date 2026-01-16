#!/usr/bin/env python3
"""
Script to rename existing crop files to add video name prefix.
Renames files in crops/ subdirectories from:
  frame{num}_class{class}_bbox{idx}.png
to:
  {video_name}_frame{num}_class{class}_bbox{idx}.png
"""

import os
import sys
from pathlib import Path
import re


def rename_crops_in_directory(crops_dir):
    """Rename all crop files in a directory to include the video name prefix."""

    # Get the video name from the parent directory (two levels up from crops/)
    video_name = crops_dir.parent.name

    # Pattern to match existing crop filenames
    pattern = re.compile(r"^(frame\d+_class\d+_bbox\d+)\.(png|txt)$")

    renamed_count = 0
    skipped_count = 0

    for file_path in crops_dir.iterdir():
        if file_path.is_file():
            filename = file_path.name
            match = pattern.match(filename)

            if match:
                # File matches the pattern and needs renaming
                base_part = match.group(1)
                extension = match.group(2)

                new_filename = f"{video_name}_{base_part}.{extension}"
                new_path = file_path.parent / new_filename

                if new_path.exists():
                    print(f"    Skipping {filename} - destination already exists")
                    skipped_count += 1
                else:
                    file_path.rename(new_path)
                    print(f"    Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
            elif filename.startswith(video_name + "_"):
                # File already has the video name prefix
                skipped_count += 1
            else:
                print(f"    Warning: Unexpected filename format: {filename}")

    return renamed_count, skipped_count


def process_directory(base_dir):
    """Process all video directories in a base directory."""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    print(f"\nProcessing directory: {base_dir}")

    total_renamed = 0
    total_skipped = 0

    # Find all crops directories
    for video_dir in base_path.iterdir():
        if video_dir.is_dir():
            crops_dir = video_dir / "crops"
            if crops_dir.exists() and crops_dir.is_dir():
                print(f"\n  Processing: {video_dir.name}/crops/")
                renamed, skipped = rename_crops_in_directory(crops_dir)
                total_renamed += renamed
                total_skipped += skipped
                print(f"    Summary: {renamed} renamed, {skipped} skipped")

    print(f"\nTotal files renamed: {total_renamed}")
    print(f"Total files skipped: {total_skipped}")


def main():
    if len(sys.argv) > 1:
        # Process specific directories from command line
        for directory in sys.argv[1:]:
            process_directory(directory)
    else:
        print("Usage: python rename_crop_files.py <directory1> [directory2] ...")
        print("Example: python rename_crop_files.py /path/to/train_data /path/to/val_data")
        sys.exit(1)


if __name__ == "__main__":
    main()
