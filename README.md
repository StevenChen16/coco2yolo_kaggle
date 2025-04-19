# COCO2YOLO Kaggle

A tool for converting between COCO and YOLO formats in Kaggle environment. This tool is based on ultralytics/JSON2YOLO and optimized for Kaggle environment, with added support for reverse conversion (YOLO to COCO).

English | [中文](README.zh.md)

## Features

- Bi-directional conversion between COCO and YOLO formats:
  - COCO JSON annotations to YOLO format
  - YOLO annotations to COCO JSON format
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

### COCO to YOLO Conversion

#### Convert Labels Only (Default Mode)

```bash
coco2yolo-kaggle coco2yolo --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo
```

By default, this will map the 91 COCO classes to 80 classes. If you want to keep the original 91 classes, use:

```bash
coco2yolo-kaggle coco2yolo --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --no-cls91to80
```

#### Copy Files Only (with Custom Directory Names)

```bash
coco2yolo-kaggle coco2yolo --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val
```

#### Complete Process with Custom Directory Names

```bash
coco2yolo-kaggle coco2yolo --mode=all --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val
```

### YOLO to COCO Conversion

#### Convert Labels Only Using Class Names

```bash
coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --class-names="person,car,bicycle"
```

#### Convert Labels Only Using YAML File

```bash
coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --yaml-path=/kaggle/input/yolo-dataset/data.yaml
```

#### Complete Process with Copying Images

```bash
coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --final-dest=/kaggle/tmp/COCO --yaml-path=/kaggle/input/yolo-dataset/data.yaml --copy-images
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

### COCO to YOLO Example

```python
# 1. Install the package
!pip install coco2yolo-kaggle

# 2. Convert labels only (with class mapping)
!coco2yolo-kaggle coco2yolo --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo

# 3. Copy dataset files with custom directory names
!coco2yolo-kaggle coco2yolo --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val

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

### YOLO to COCO Example

```python
# 1. Install the package
!pip install coco2yolo-kaggle

# 2. Convert YOLO labels to COCO format using a YAML file
!coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --yaml-path=/kaggle/input/yolo-dataset/data.yaml

# 3. Python API for full control
from coco2yolo_kaggle import convert_yolo_dataset

# Convert YOLO dataset to COCO with more options
result = convert_yolo_dataset(
    yolo_dataset_path='/kaggle/input/yolo-dataset',
    output_dir='/kaggle/working/yolo_coco',
    final_dest='/kaggle/tmp/COCO',
    class_names=['person', 'car', 'bicycle'],  # Explicit class names
    split_names=('train', 'val'),              # Dataset splits to process
    image_ext='jpg',                           # Image file extension
    copy_images=True,                          # Copy images to final destination
    max_workers=8                              # Parallel processing threads
)

print(f"Generated annotation files: {result}")
```

## Acknowledgements

This tool is based on [ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO), thanks to the original authors' contributions.