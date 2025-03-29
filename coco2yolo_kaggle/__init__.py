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
    max_workers=8
):
    """
    Copy dataset files to final destination
    
    Parameters:
        json_dir: COCO JSON annotation directory (needed to locate image files)
        output_dir: Directory with converted YOLO labels
        final_dest: Final dataset directory
        max_workers: Number of parallel worker threads
    
    Returns:
        Path to the final destination directory
    """
    import os
    
    # Create target directory structure
    os.makedirs(f'{final_dest}/images', exist_ok=True)
    os.makedirs(f"{final_dest}/labels", exist_ok=True)
    
    # Copy image files
    copy_tasks = [
        (f"{'/'.join(json_dir.split('/')[:-1])}/train2017", f"{final_dest}/images/train2017"),
        (f"{'/'.join(json_dir.split('/')[:-1])}/val2017", f"{final_dest}/images/val2017")
    ]
    
    for src, dest in copy_tasks:
        copy_dir_parallel(src, dest, max_workers=max_workers)
    
    # Move label files - using parallel processing
    move_tasks = [
        (f"{output_dir}/labels/train", f"{final_dest}/labels/train2017"),
        (f"{output_dir}/labels/val", f"{final_dest}/labels/val2017")
    ]
    
    # Using ProcessPoolExecutor for parallel move operations
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        move_results = list(executor.map(move_dir, move_tasks))
    
    for result in move_results:
        print(result)
        
    print(f"Dataset preparation complete! Location: {final_dest}")
    print(f"- Images: {final_dest}/images/")
    print(f"- Labels: {final_dest}/labels/")
    
    return final_dest

def convert_coco_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    use_segments=True,
    cls91to80=True,
    max_workers=8,
    copy_files=True
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
            max_workers=max_workers
        )
    else:
        print(f"Labels conversion complete! Location: {output_dir}")
        print(f"- Labels: {output_dir}/labels/")
        return output_dir