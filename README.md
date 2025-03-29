# COCO2YOLO Kaggle

A tool for converting COCO dataset to YOLO format in Kaggle environment. This tool is based on ultralytics/JSON2YOLO and optimized for Kaggle environment.

## Features

- Automatically convert COCO JSON annotations to YOLO format
- Intelligently handle Kaggle storage space limitations
- Support parallel file operations for faster processing
- Support segment annotations
- Flexible commands for different operations

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
    cls91to80=True
)
```

#### File Copy Only

```python
from coco2yolo_kaggle import copy_dataset

# Copy dataset files to final destination
dataset_path = copy_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    max_workers=8
)
```

#### Complete Conversion Process

```python
from coco2yolo_kaggle import convert_coco_dataset

# Complete process (both conversion and copying)
dataset_path = convert_coco_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    use_segments=True,
    cls91to80=True,
    max_workers=8,
    copy_files=True  # Set to False to skip copying files
)
```

### Command Line Usage

#### Convert Labels Only (Default Mode)

```bash
coco2yolo-kaggle --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo
```

#### Copy Files Only

```bash
coco2yolo-kaggle --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017
```

#### Complete Process

```bash
coco2yolo-kaggle --mode=all --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017
```

## Kaggle Example Code

```python
# 1. Install the package
!pip install coco2yolo-kaggle

# 2. Convert labels only
!coco2yolo-kaggle --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo

# 3. Copy dataset files (if needed)
!coco2yolo-kaggle --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017

# 4. Train a YOLOv8 model
!pip install ultralytics
from ultralytics import YOLO

# Create dataset configuration file
%%writefile coco.yaml
path: /kaggle/tmp/COCO2017
train: images/train2017
val: images/val2017
nc: 80
names: ['person', 'bicycle', 'car', ... ] # Complete 80 class names

# Train the model
model = YOLO('yolov8n.pt')  # Use nano model
results = model.train(data='coco.yaml', epochs=3, imgsz=640)
```

## Acknowledgements

This tool is based on [ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO), thanks to the original authors' contributions.