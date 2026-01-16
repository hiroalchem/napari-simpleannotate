#!/usr/bin/env python3
"""
Script to crop existing YOLO annotations and images.
Based on the crop functionality from BboxVideoQWidget.
"""

import os
import argparse
from pathlib import Path
import yaml
from skimage import io
import numpy as np


def load_yolo_annotations(txt_path, img_width, img_height):
    """Load YOLO format annotations and convert to pixel coordinates."""
    annotations = []
    
    if not os.path.exists(txt_path):
        return annotations
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx = float(parts[1]) * img_width
                cy = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                
                annotations.append({
                    'class': class_id,
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2)
                })
    
    return annotations


def crop_and_save(input_dir, output_dir, crop_width, crop_height):
    """Crop images and annotations based on bounding box centers."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    crops_dir = output_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy class.yaml
    class_yaml_src = input_path / "class.yaml"
    if class_yaml_src.exists():
        class_yaml_dst = output_path / "class.yaml"
        with open(class_yaml_src, 'r') as f:
            class_data = yaml.safe_load(f)
        with open(class_yaml_dst, 'w') as f:
            yaml.dump(class_data, f)
        print(f"Copied class.yaml to {class_yaml_dst}")
    
    # Track processed bboxes across all images
    total_processed = 0
    
    # Process each image
    for img_path in sorted(input_path.glob("*.png")):
        base_name = img_path.stem
        txt_path = input_path / f"{base_name}.txt"
        
        if not txt_path.exists():
            continue
        
        print(f"\nProcessing {img_path.name}...")
        
        # Load image
        image = io.imread(img_path)
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        annotations = load_yolo_annotations(txt_path, img_width, img_height)
        
        if not annotations:
            continue
        
        # Track which bboxes have been processed for this image
        processed_indices = set()
        
        for i, bbox in enumerate(annotations):
            if i in processed_indices:
                continue
            
            # Calculate crop center
            cx = (bbox['x1'] + bbox['x2']) // 2
            cy = (bbox['y1'] + bbox['y2']) // 2
            
            # Calculate crop boundaries
            crop_x1 = cx - crop_width // 2
            crop_y1 = cy - crop_height // 2
            crop_x2 = crop_x1 + crop_width
            crop_y2 = crop_y1 + crop_height
            
            # Check if bbox crosses crop boundary (skip if it does)
            rel_x1 = bbox['x1'] - crop_x1
            rel_y1 = bbox['y1'] - crop_y1
            rel_x2 = bbox['x2'] - crop_x1
            rel_y2 = bbox['y2'] - crop_y1
            
            if (rel_x1 < 0 or rel_y1 < 0 or 
                rel_x2 > crop_width or rel_y2 > crop_height):
                print(f"  Skipping bbox {i} (class {bbox['class']}) - crosses crop boundary")
                continue
            
            # Adjust crop boundaries to fit within image
            if crop_x1 < 0:
                crop_x2 -= crop_x1
                crop_x1 = 0
            if crop_y1 < 0:
                crop_y2 -= crop_y1
                crop_y1 = 0
            if crop_x2 > img_width:
                crop_x1 -= (crop_x2 - img_width)
                crop_x2 = img_width
            if crop_y2 > img_height:
                crop_y1 -= (crop_y2 - img_height)
                crop_y2 = img_height
            
            # Ensure crop is still valid
            if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_width or crop_y2 > img_height:
                print(f"  Skipping bbox {i} (class {bbox['class']}) - cannot fit crop in image")
                continue
            
            # Extract crop
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Save crop with directory name prefix
            dir_name = input_path.name  # Get the directory name
            crop_filename = f"{dir_name}_{base_name}_class{bbox['class']}_bbox{i}.png"
            crop_path = crops_dir / crop_filename
            io.imsave(crop_path, crop)
            print(f"  Saved crop: {crop_filename}")
            
            # Mark this bbox as processed
            processed_indices.add(i)
            total_processed += 1
            
            # Find all bboxes contained in this crop
            contained_bboxes = []
            for j, other_bbox in enumerate(annotations):
                if (other_bbox['x1'] >= crop_x1 and other_bbox['y1'] >= crop_y1 and
                    other_bbox['x2'] <= crop_x2 and other_bbox['y2'] <= crop_y2):
                    contained_bboxes.append((j, other_bbox))
                    if j != i:
                        processed_indices.add(j)
                        print(f"    Bbox {j} (class {other_bbox['class']}) is contained in crop")
            
            # Save annotation file for this crop
            annotation_filename = crop_filename.replace(".png", ".txt")
            annotation_path = crops_dir / annotation_filename
            
            with open(annotation_path, 'w') as f:
                for idx, contained_bbox in contained_bboxes:
                    # Convert to YOLO format relative to crop
                    rel_cx = ((contained_bbox['x1'] + contained_bbox['x2']) / 2 - crop_x1) / crop_width
                    rel_cy = ((contained_bbox['y1'] + contained_bbox['y2']) / 2 - crop_y1) / crop_height
                    rel_w = (contained_bbox['x2'] - contained_bbox['x1']) / crop_width
                    rel_h = (contained_bbox['y2'] - contained_bbox['y1']) / crop_height
                    
                    f.write(f"{contained_bbox['class']} {rel_cx:.6f} {rel_cy:.6f} {rel_w:.6f} {rel_h:.6f}\n")
    
    print(f"\nTotal cropped annotations: {total_processed}")


def main():
    parser = argparse.ArgumentParser(description='Crop existing YOLO annotations and images')
    parser.add_argument('input_dir', help='Directory containing images and YOLO annotations')
    parser.add_argument('output_dir', help='Output directory for cropped images and annotations')
    parser.add_argument('--width', type=int, default=256, help='Crop width in pixels (default: 256)')
    parser.add_argument('--height', type=int, default=256, help='Crop height in pixels (default: 256)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    print(f"Cropping annotations from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Crop size: {args.width}x{args.height}")
    
    crop_and_save(args.input_dir, args.output_dir, args.width, args.height)


if __name__ == "__main__":
    main()