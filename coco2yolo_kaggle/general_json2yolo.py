import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
import glob
import shutil
from PIL import Image, ExifTags

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break

def exif_size(img):
    """Returns the EXIF-corrected PIL image size as a tuple (width, height)."""
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270
            s = (s[1], s[0])
    except Exception:
        pass

    return s

def make_dirs(dir="new_dir/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "train", dir / "labels" / "train", dir / "labels" / "val":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def coco91_to_coco80_class():
    """Map COCO 91 classes to 80 classes"""
    return [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
        None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
        None, 73, 74, 75, 76, 77, 78, 79, None
    ]

def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    """Splits array indices for train, test, and validate datasets according to specified ratios."""
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

def split_files(out_path, file_name, prefix_path=""):  # split training data
    """Splits file names into separate train, test, and val datasets and writes them to prefixed paths."""
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=0.9, test=0.1, validate=0.0)
    datasets = {"train": i, "test": j, "val": k}
    for key, item in datasets.items():
        if item.any():
            with open(f"{out_path}_{key}.txt", "a") as file:
                for i in item:
                    file.write(f"{prefix_path}{file_name[i]}\n")

def write_data_data(fname="data.data", nc=80):
    """Writes a Darknet-style .data file with dataset and training configuration."""
    lines = [
        f"classes = {nc:g}\n",
        "train =../out/data_train.txt\n",
        "valid =../out/data_test.txt\n",
        "names =../out/data.names\n",
        "backup = backup/\n",
        "eval = coco\n",
    ]

    with open(fname, "a") as f:
        f.writelines(lines)

def convert_coco_json(json_dir="../coco/annotations/", output_dir="output", use_segments=False, cls91to80=False):
    """
    Convert COCO JSON format to YOLO txt format
    
    Parameters:
        json_dir: COCO annotation directory
        output_dir: Output directory
        use_segments: Whether to use segment annotations
        cls91to80: Whether to map 91 classes to 80 classes
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(exist_ok=True)
    coco80 = coco91_to_coco80_class()
    
    # Create separate directories for train and val labels
    (save_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (save_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Import JSON files
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        # Skip non-instance annotation files
        if 'instances' not in json_file.stem:
            print(f"Skipping {json_file} - not an instance annotation file")
            continue
            
        print(f"Processing {json_file}...")
        
        # Determine if this is train or val based on filename
        if 'train' in json_file.stem:
            save_path = save_dir / "labels" / "train"
        elif 'val' in json_file.stem:
            save_path = save_dir / "labels" / "val"
        else:
            print(f"Cannot determine if {json_file} is train or val, skipping")
            continue
            
        with open(json_file) as f:
            data = json.load(f)

        # Process annotation information
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write label files
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotating {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                
                if "bbox" not in ann:
                    continue
                
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # Convert [x,y,w,h] to [cx,cy,w,h]
                box[[0, 2]] /= w  # Normalize
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:  # Check for valid box
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1
                if cls is None:  # Category filtering
                    continue
                
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                
                # Process segment annotations
                if use_segments and "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):
                        if len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        if s not in segments:
                            segments.append(s)
                    else:
                        # RLE format segments - not processed
                        pass

            # Save to appropriate train or val directory
            with open((save_path / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments and i < len(segments) else bboxes[i]),)
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

    print("Conversion complete!")
    print(f"Train labels saved at: {save_dir}/labels/train/")
    print(f"Val labels saved at: {save_dir}/labels/val/")