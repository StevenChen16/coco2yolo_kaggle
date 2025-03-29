from .general_json2yolo import convert_coco_json, coco91_to_coco80_class
from .file_utils import copy_dir_parallel, move_dir

import concurrent.futures

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