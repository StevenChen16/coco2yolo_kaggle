# COCO2YOLO Kaggle

A tool for converting COCO dataset to YOLO format in Kaggle environment. This tool is based on ultralytics/JSON2YOLO and optimized for Kaggle environment.

## Features

- Automatically convert COCO JSON annotations to YOLO format
- Intelligently handle Kaggle storage space limitations
- Support parallel file operations for faster processing
- Support segment annotations
- Flexible commands for different operations
- Customizable directory structure

## Installation

```bash
pip install coco2yolo-kaggle
```

## Usage

### Python API

#### Label Conversion Only

```python
from coco2yolo_kaggle import convert_coco_labels

# Convert labels only
labels_dir = convert_coco_labels(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    use_segments=True,
    cls91to80=True  # Map 91 COCO classes to 80 classes (default: True)
)
```

#### File Copy Only (with Custom Directory Names)

```python
from coco2yolo_kaggle import copy_dataset

# Copy dataset files to final destination
dataset_path = copy_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    max_workers=8,
    train_dir_name="train",  # Custom train directory name (default: "train2017")
    val_dir_name="val"       # Custom validation directory name (default: "val2017")
)
```

#### Complete Conversion Process

```python
from coco2yolo_kaggle import convert_coco_dataset

# Complete process with custom directory names
dataset_path = convert_coco_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    use_segments=True,
    cls91to80=True,  # Map 91 COCO classes to 80 classes (default: True)
    max_workers=8,
    copy_files=True,  # Set to False to skip copying files
    train_dir_name="train",  # Custom destination directory (default: "train2017")
    val_dir_name="val",      # Custom destination directory (default: "val2017")
    src_train_dir_name="train2017",  # Source train directory name
    src_val_dir_name="val2017"       # Source validation directory name
)
```

### Command Line Usage

#### Convert Labels Only (Default Mode)

```bash
coco2yolo-kaggle --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo
```

By default, this will map the 91 COCO classes to 80 classes. If you want to keep the original 91 classes, use:

```bash
coco2yolo-kaggle --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --no-cls91to80
```

#### Copy Files Only (with Custom Directory Names)

```bash
coco2yolo-kaggle --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val
```

#### Complete Process with Custom Directory Names

```bash
coco2yolo-kaggle --mode=all --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val
```

## Class Mapping and Customizing Directory Structure

### Class Mapping
- By default, the tool maps the original 91 COCO classes to the standard 80 classes (`cls91to80=True`)
- To preserve the original 91 classes, use the `--no-cls91to80` flag

### Directory Name Customization
You can customize the directory names using these parameters:

- `--train-dir-name`: Set destination train directory name (default: "train2017")
- `--val-dir-name`: Set destination validation directory name (default: "val2017")
- `--src-train-dir`: Source train directory name (default: "train2017")
- `--src-val-dir`: Source validation directory name (default: "val2017")

## Kaggle Example Code

```python
# 1. Install the package
!pip install coco2yolo-kaggle

# 2. Convert labels only (with class mapping)
!coco2yolo-kaggle --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo

# 3. Copy dataset files with custom directory names
!coco2yolo-kaggle --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val

# 4. Train a YOLOv8 model
!pip install ultralytics
from ultralytics import YOLO

# Create dataset configuration file with custom directory structure
%%writefile coco.yaml
path: /kaggle/tmp/COCO2017
train: images/train  # Using custom directory name
val: images/val      # Using custom directory name
nc: 80
names: ['person', 'bicycle', 'car', ... ] # Complete 80 class names

# Train the model
model = YOLO('yolov8n.pt')  # Use nano model
results = model.train(data='coco.yaml', epochs=3, imgsz=640)
```

## Acknowledgements

This tool is based on [ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO), thanks to the original authors' contributions.