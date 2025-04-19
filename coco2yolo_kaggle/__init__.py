from .general_json2yolo import convert_coco_json, coco91_to_coco80_class
from .file_utils import copy_dir_parallel, move_dir
from .yolo2coco import yolo_to_coco, yolo_to_coco_from_yaml

import concurrent.futures
import shutil

__version__ = "0.1.0"

def convert_coco_labels(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    use_segments=True,
    cls91to80=True
):
    """
    Convert COCO JSON annotations to YOLO format
    
    Parameters:
        json_dir: COCO JSON annotation directory
        output_dir: Output directory for YOLO labels
        use_segments: Whether to use segment annotations
        cls91to80: Whether to map 91 classes to 80 classes
        
    Returns:
        Path to the output directory
    """
    # Convert COCO JSON to YOLO format
    convert_coco_json(
        json_dir=json_dir,
        output_dir=output_dir,
        use_segments=use_segments,
        cls91to80=cls91to80
    )
    
    return output_dir

def copy_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    max_workers=8,
    train_dir_name="train2017",
    val_dir_name="val2017",
    src_train_dir_name="train2017",
    src_val_dir_name="val2017"
):
    """
    Copy dataset files to final destination
    
    Parameters:
        json_dir: COCO JSON annotation directory (needed to locate image files)
        output_dir: Directory with converted YOLO labels
        final_dest: Final dataset directory
        max_workers: Number of parallel worker threads
        train_dir_name: Destination train subdirectory name (default: "train2017")
        val_dir_name: Destination validation subdirectory name (default: "val2017")
        src_train_dir_name: Source train subdirectory name (default: "train2017")
        src_val_dir_name: Source validation subdirectory name (default: "val2017")
    
    Returns:
        Path to the final destination directory
    """
    import os
    
    # Create target directory structure
    os.makedirs(f'{final_dest}/images', exist_ok=True)
    os.makedirs(f"{final_dest}/labels", exist_ok=True)
    
    # Get base directory from json_dir
    base_dir = '/'.join(json_dir.split('/')[:-1])
    
    # Copy image files
    copy_tasks = [
        (f"{base_dir}/{src_train_dir_name}", f"{final_dest}/images/{train_dir_name}"),
        (f"{base_dir}/{src_val_dir_name}", f"{final_dest}/images/{val_dir_name}")
    ]
    
    for src, dest in copy_tasks:
        copy_dir_parallel(src, dest, max_workers=max_workers)
    
    # Move label files - using parallel processing
    move_tasks = [
        (f"{output_dir}/labels/train", f"{final_dest}/labels/{train_dir_name}"),
        (f"{output_dir}/labels/val", f"{final_dest}/labels/{val_dir_name}")
    ]
    
    # Using ProcessPoolExecutor for parallel move operations
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        move_results = list(executor.map(move_dir, move_tasks))
    
    for result in move_results:
        print(result)
        
    print(f"Dataset preparation complete! Location: {final_dest}")
    print(f"- Images: {final_dest}/images/{train_dir_name} and {final_dest}/images/{val_dir_name}")
    print(f"- Labels: {final_dest}/labels/{train_dir_name} and {final_dest}/labels/{val_dir_name}")
    
    return final_dest

def convert_coco_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    use_segments=True,
    cls91to80=True,
    max_workers=8,
    copy_files=True,
    train_dir_name="train2017",
    val_dir_name="val2017",
    src_train_dir_name="train2017",
    src_val_dir_name="val2017"
):
    """
    Execute the complete COCO dataset conversion process:
    1. Convert COCO JSON annotations to YOLO format
    2. Optionally copy image files and move label files to final location
    
    Parameters:
        json_dir: COCO JSON annotation directory
        output_dir: Temporary output directory
        final_dest: Final dataset directory
        use_segments: Whether to use segment annotations
        cls91to80: Whether to map 91 classes to 80 classes
        max_workers: Number of parallel worker threads
        copy_files: Whether to copy image files and move labels to final destination
        train_dir_name: Destination train subdirectory name (default: "train2017")
        val_dir_name: Destination validation subdirectory name (default: "val2017")
        src_train_dir_name: Source train subdirectory name (default: "train2017")
        src_val_dir_name: Source validation subdirectory name (default: "val2017")
    
    Returns:
        Path to the resulting dataset directory
    """
    # 1. Convert labels
    convert_coco_labels(
        json_dir=json_dir,
        output_dir=output_dir,
        use_segments=use_segments,
        cls91to80=cls91to80
    )
    
    # 2. Optionally copy files
    if copy_files:
        return copy_dataset(
            json_dir=json_dir,
            output_dir=output_dir,
            final_dest=final_dest,
            max_workers=max_workers,
            train_dir_name=train_dir_name,
            val_dir_name=val_dir_name,
            src_train_dir_name=src_train_dir_name,
            src_val_dir_name=src_val_dir_name
        )
    else:
        print(f"Labels conversion complete! Location: {output_dir}")
        print(f"- Labels: {output_dir}/labels/")
        return output_dir


def convert_yolo_labels(
    yolo_dataset_path="/kaggle/input/yolo-dataset",
    output_dir="/kaggle/working/yolo_coco",
    class_names=None,
    yaml_path=None,
    split_names=("train", "val"),
    image_ext="jpg"
):
    """
    Convert YOLO format annotations to COCO format
    
    Parameters:
        yolo_dataset_path: Path to YOLO dataset with labels/{train,val} and images/{train,val} structure
        output_dir: Output directory for COCO annotations
        class_names: List of class names or dict {class_id: class_name} (required if yaml_path not provided)
        yaml_path: Path to YAML file with class names (alternative to class_names)
        split_names: Dataset splits to process (default: ["train", "val"])
        image_ext: Image file extension (default: "jpg")
        
    Returns:
        Dictionary with paths to generated annotation files
    """
    if yaml_path:
        return yolo_to_coco_from_yaml(
            yolo_dataset_path=yolo_dataset_path,
            output_dir=output_dir,
            yaml_path=yaml_path,
            split_names=split_names,
            image_ext=image_ext,
            copy_images=False
        )
    elif class_names:
        return yolo_to_coco(
            yolo_dataset_path=yolo_dataset_path,
            output_dir=output_dir,
            class_names=class_names,
            split_names=split_names,
            image_ext=image_ext,
            copy_images=False
        )
    else:
        raise ValueError("Either class_names or yaml_path must be provided")


def convert_yolo_dataset(
    yolo_dataset_path="/kaggle/input/yolo-dataset",
    output_dir="/kaggle/working/yolo_coco",
    final_dest="/kaggle/tmp/COCO2017",
    class_names=None,
    yaml_path=None,
    split_names=("train", "val"),
    image_ext="jpg",
    copy_images=True,
    max_workers=8
):
    """
    Execute the complete YOLO to COCO dataset conversion process:
    1. Convert YOLO labels to COCO JSON format
    2. Optionally copy image files to final destination
    
    Parameters:
        yolo_dataset_path: Path to YOLO dataset with labels/{train,val} and images/{train,val} structure
        output_dir: Temporary output directory for annotations
        final_dest: Final dataset directory
        class_names: List of class names or dict {class_id: class_name} (required if yaml_path not provided)
        yaml_path: Path to YAML file with class names (alternative to class_names)
        split_names: Dataset splits to process (default: ["train", "val"])
        image_ext: Image file extension (default: "jpg")
        copy_images: Whether to copy images to final destination
        max_workers: Number of parallel worker threads
    
    Returns:
        Dictionary with paths to generated files
    """
    # 1. Convert labels
    result = convert_yolo_labels(
        yolo_dataset_path=yolo_dataset_path,
        output_dir=output_dir,
        class_names=class_names,
        yaml_path=yaml_path,
        split_names=split_names,
        image_ext=image_ext
    )
    
    # 2. Optionally copy images
    if copy_images:
        # Create destination directories
        import os
        os.makedirs(final_dest, exist_ok=True)
        images_dest = os.path.join(final_dest, "images")
        os.makedirs(images_dest, exist_ok=True)
        
        # Copy images for each split
        for split in split_names:
            src_images = os.path.join(yolo_dataset_path, "images", split)
            dst_images = os.path.join(images_dest, split)
            
            if os.path.exists(src_images):
                copy_dir_parallel(src_images, dst_images, max_workers=max_workers)
                print(f"Copied images from {src_images} to {dst_images}")
            else:
                print(f"Warning: Source images directory {src_images} not found, skipping copy")
        
        # Copy annotations
        annotations_dir = os.path.join(final_dest, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        for split, annotation_file in result.items():
            dst_file = os.path.join(annotations_dir, os.path.basename(annotation_file))
            shutil.copy2(annotation_file, dst_file)
            result[split] = dst_file
            print(f"Copied annotation file to {dst_file}")
        
        print(f"Dataset preparation complete! Location: {final_dest}")
        print(f"- Images: {images_dest}")
        print(f"- Annotations: {annotations_dir}")
    
    return result