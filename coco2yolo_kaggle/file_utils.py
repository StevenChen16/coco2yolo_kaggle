import os
import shutil
import concurrent.futures
from tqdm import tqdm
import os.path as osp

def get_all_files(directory):
    """Get paths of all files in a directory"""
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def copy_single_file(args):
    """Copy a single file"""
    src_file, dest_file = args
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    shutil.copy2(src_file, dest_file)
    return src_file

def copy_dir_parallel(src, dest, max_workers=4):
    """Copy all files in a directory in parallel with progress bar"""
    # Get all source files
    src_files = get_all_files(src)
    total_files = len(src_files)
    
    # Create corresponding destination file paths
    copy_tasks = []
    for src_file in src_files:
        rel_path = os.path.relpath(src_file, src)
        dest_file = os.path.join(dest, rel_path)
        copy_tasks.append((src_file, dest_file))
    
    # Use thread pool to copy files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create progress bar
        with tqdm(total=total_files, desc=f"Copying {osp.basename(src)}") as pbar:
            # Submit all copy tasks
            futures = [executor.submit(copy_single_file, task) for task in copy_tasks]
            
            # Wait for tasks to complete and update progress bar
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"Error copying file: {e}")
    
    return f"Completed copying {src} to {dest}, total {total_files} files"

def move_dir(src_dest):
    """Move directory"""
    src, dest = src_dest
    print(f"Moving {src} to {dest}")
    if os.path.exists(src):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        return f"Completed moving {src}"
    else:
        print(f"Source directory {src} does not exist")
        return f"Move failed: {src} does not exist"