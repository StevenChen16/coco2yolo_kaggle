import os
import json
import glob
from PIL import Image
import numpy as np
from pathlib import Path

def yolo_to_coco(
    yolo_dataset_path,
    output_dir,
    class_names,
    split_names=("train", "val"),
    image_ext="jpg",
    copy_images=False
):
    """
    Convert YOLO format dataset to COCO format
    
    Args:
        yolo_dataset_path: Path to YOLO dataset
        output_dir: Output directory for COCO format data
        class_names: List of class names or dict {class_id: class_name}
        split_names: Dataset split names (default: ["train", "val"])
        image_ext: Image file extension (default: "jpg")
        copy_images: Whether to copy images to output directory (default: False)
    
    Returns:
        Dictionary with paths to generated annotation files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Prepare class dictionary
    if isinstance(class_names, dict):
        class_dict = {int(k): v for k, v in class_names.items()}
    else:
        class_dict = {i: name for i, name in enumerate(class_names)}
    
    result_files = {}
    
    # Process each split (train/val)
    for split in split_names:
        # Define paths
        labels_path = os.path.join(yolo_dataset_path, "labels", split)
        images_path = os.path.join(yolo_dataset_path, "images", split)
        
        if not os.path.exists(labels_path):
            print(f"Warning: {labels_path} does not exist, skipping {split} split")
            continue
        
        if not os.path.exists(images_path):
            print(f"Warning: {images_path} does not exist, skipping {split} split")
            continue
        
        # Initialize COCO JSON structure
        coco_json = {
            "info": {
                "description": f"Converted from YOLO format - {split} set",
                "url": "",
                "version": "1.0",
                "year": 2024,
                "contributor": "coco2yolo_kaggle",
                "date_created": ""
            },
            "licenses": [
                {
                    "url": "",
                    "id": 1,
                    "name": "Unknown"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for class_id, class_name in class_dict.items():
            coco_json["categories"].append({
                "supercategory": "none",
                "id": class_id + 1,  # COCO IDs start from 1
                "name": class_name
            })
        
        # Create output image directory if copying
        if copy_images:
            out_images_dir = os.path.join(output_dir, split)
            os.makedirs(out_images_dir, exist_ok=True)
        
        # Process each label file
        annotation_id = 1
        for label_file in glob.glob(os.path.join(labels_path, "*.txt")):
            base_name = os.path.basename(label_file)
            image_name = os.path.splitext(base_name)[0] + f".{image_ext}"
            image_path = os.path.join(images_path, image_name)
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping label {label_file}")
                continue
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue
            
            # Add image to COCO JSON
            image_id = len(coco_json["images"]) + 1
            coco_json["images"].append({
                "license": 1,
                "file_name": image_name,
                "height": height,
                "width": width,
                "id": image_id
            })
            
            # Copy image if requested
            if copy_images:
                dst_image = os.path.join(out_images_dir, image_name)
                if not os.path.exists(dst_image):
                    try:
                        shutil.copy2(image_path, dst_image)
                    except Exception as e:
                        print(f"Error copying image {image_path}: {e}")
            
            # Process annotations
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:  # Class + 4 coords (bbox)
                            continue
                        
                        class_id = int(parts[0])
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                        
                        # Convert YOLO normalized coordinates to COCO pixel coordinates
                        x_min = int((x_center - bbox_width / 2) * width)
                        y_min = int((y_center - bbox_height / 2) * height)
                        bbox_width_px = int(bbox_width * width)
                        bbox_height_px = int(bbox_height * height)
                        
                        # Ensure coordinates are valid
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        bbox_width_px = min(bbox_width_px, width - x_min)
                        bbox_height_px = min(bbox_height_px, height - y_min)
                        
                        # Add annotation to COCO JSON
                        coco_json["annotations"].append({
                            "segmentation": [],
                            "area": bbox_width_px * bbox_height_px,
                            "iscrowd": 0,
                            "image_id": image_id,
                            "bbox": [x_min, y_min, bbox_width_px, bbox_height_px],
                            "category_id": class_id + 1,  # COCO IDs start from 1
                            "id": annotation_id
                        })
                        annotation_id += 1
                        
                        # Handle segmentation if available (parts beyond the bbox)
                        if len(parts) > 5:
                            # Parse segmentation points (x1, y1, x2, y2, ...)
                            segment_points = list(map(float, parts[5:]))
                            # Convert normalized coordinates to pixel coordinates
                            segment_px = []
                            for i in range(0, len(segment_points), 2):
                                if i + 1 < len(segment_points):
                                    x = int(segment_points[i] * width)
                                    y = int(segment_points[i + 1] * height)
                                    segment_px.extend([x, y])
                            
                            if segment_px:
                                coco_json["annotations"][-1]["segmentation"] = [segment_px]
            
            except Exception as e:
                print(f"Error processing label file {label_file}: {e}")
        
        # Save COCO JSON file
        output_json = os.path.join(annotations_dir, f"instances_{split}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_json, f, ensure_ascii=False, indent=4)
        
        result_files[split] = output_json
        print(f"Created COCO annotations for {split} split: {output_json}")
    
    return result_files

def yolo_to_coco_from_yaml(
    yolo_dataset_path,
    output_dir,
    yaml_path,
    split_names=("train", "val"),
    image_ext="jpg",
    copy_images=False
):
    """
    Convert YOLO format dataset to COCO format using a YAML config file
    
    Args:
        yolo_dataset_path: Path to YOLO dataset
        output_dir: Output directory for COCO format data
        yaml_path: Path to YAML file containing class names
        split_names: Dataset split names (default: ["train", "val"])
        image_ext: Image file extension (default: "jpg")
        copy_images: Whether to copy images to output directory (default: False)
    
    Returns:
        Dictionary with paths to generated annotation files
    """
    import yaml
    
    # Read YAML file
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # Get class names
    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError(f"No class names found in YAML file: {yaml_path}")
    
    return yolo_to_coco(
        yolo_dataset_path=yolo_dataset_path,
        output_dir=output_dir,
        class_names=class_names,
        split_names=split_names,
        image_ext=image_ext,
        copy_images=copy_images
    )
