#!/usr/bin/env python3
"""
Script to find and optionally delete empty YOLO annotation files.
This helps identify crops that need to be regenerated due to the bug.
"""

import os
import argparse
from pathlib import Path


def find_empty_annotations(root_dir, delete=False, all_txt=False):
    """Find empty annotation files in crops directories or all .txt files.

    Args:
        root_dir: Root directory to search
        delete: Whether to delete empty files
        all_txt: If True, search all .txt files recursively.
                 If False, only search in 'crops/' subdirectories.
    """

    root_path = Path(root_dir)
    empty_files = []
    empty_dirs = set()

    # Find all .txt files
    if all_txt:
        # Search all .txt files recursively
        pattern = "**/*.txt"
    else:
        # Only search in crops subdirectories
        pattern = "crops/*.txt"

    for txt_path in root_path.rglob(pattern):
        # Skip class.yaml or other config files
        if txt_path.name in ['class.yaml', 'classes.txt', 'class.txt']:
            continue

        # Check if file is empty
        if txt_path.stat().st_size == 0:
            empty_files.append(txt_path)
            # Add parent directory to affected dirs
            if 'crops' in txt_path.parts:
                # If in crops/ subdirectory, get the parent of crops/
                crops_idx = txt_path.parts.index('crops')
                parent_dir = Path(*txt_path.parts[:crops_idx])
                empty_dirs.add(parent_dir)
            else:
                # Otherwise, just use the directory containing the file
                empty_dirs.add(txt_path.parent)

    # Report findings
    print(f"Found {len(empty_files)} empty annotation files")
    print(f"Affected directories: {len(empty_dirs)}")

    if empty_files:
        print("\nEmpty annotation files:")
        for f in sorted(empty_files)[:20]:  # Show first 20
            print(f"  {f}")
        if len(empty_files) > 20:
            print(f"  ... and {len(empty_files) - 20} more")

    if empty_dirs:
        print("\nAffected video directories (need reprocessing):")
        for d in sorted(empty_dirs):
            print(f"  {d}")

    # Delete if requested
    if delete and empty_files:
        response = input(
            f"\nDelete {len(empty_files)} empty files? (yes/no): "
        )
        if response.lower() == 'yes':
            deleted_files = []
            for f in empty_files:
                # Record what we're deleting
                deleted_entry = {'annotation': str(f)}

                # Delete annotation file
                f.unlink()

                # Also delete corresponding image if exists
                img_path = f.with_suffix('.png')
                if img_path.exists():
                    img_path.unlink()
                    deleted_entry['image'] = str(img_path)
                    print(
                        f"Deleted {f.name} and {img_path.name}"
                    )
                else:
                    print(f"Deleted {f.name}")

                deleted_files.append(deleted_entry)

            # Save list of deleted files
            log_path = Path(root_dir) / "deleted_annotations.log"
            with open(log_path, 'w') as log:
                log.write(
                    f"Deleted {len(deleted_files)} empty "
                    f"annotation files\n"
                )
                log.write("=" * 80 + "\n\n")
                for entry in deleted_files:
                    log.write(f"Annotation: {entry['annotation']}\n")
                    if 'image' in entry:
                        log.write(f"Image:      {entry['image']}\n")
                    log.write("\n")

            print(
                f"\nDeleted {len(empty_files)} annotation files "
                f"and their images"
            )
            print(f"Log saved to: {log_path}")
        else:
            print("Deletion cancelled")

    return empty_files, empty_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Find empty YOLO annotation files'
    )
    parser.add_argument(
        'root_dir',
        help='Root directory to search recursively'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete empty annotation files and their images'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Search all .txt files (not just crops/ subdirectories)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"Error: Directory '{args.root_dir}' does not exist")
        return

    search_mode = (
        "all .txt files" if args.all else "crops/ subdirectories"
    )
    print(f"Searching for empty annotations in: {args.root_dir}")
    print(f"Search mode: {search_mode}")
    find_empty_annotations(args.root_dir, args.delete, args.all)


if __name__ == "__main__":
    main()
