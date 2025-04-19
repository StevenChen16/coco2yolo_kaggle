# COCO2YOLO Kaggle

一个在Kaggle环境中用于COCO和YOLO格式之间相互转换的工具。该工具基于ultralytics/JSON2YOLO，并针对Kaggle环境进行了优化，增加了反向转换（YOLO到COCO）的支持。

[English](README.md) | 中文

## 功能特点

- 支持COCO和YOLO格式的双向转换：
  - COCO JSON标注转换为YOLO格式
  - YOLO标注转换为COCO JSON格式
- 智能处理Kaggle存储空间限制
- 支持并行文件操作以加快处理速度
- 支持分割标注（segment annotations）
- 灵活的命令行操作
- 可自定义目录结构

## 安装

```bash
pip install coco2yolo-kaggle
```

## 使用方法

### Python API

#### 仅转换标签（COCO到YOLO）

```python
from coco2yolo_kaggle import convert_coco_labels

# 仅转换标签
labels_dir = convert_coco_labels(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    use_segments=True,
    cls91to80=True  # 将91个COCO类别映射到80个类别（默认：True）
)
```

#### 仅复制文件（使用自定义目录名）

```python
from coco2yolo_kaggle import copy_dataset

# 将数据集文件复制到最终目标位置
dataset_path = copy_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    max_workers=8,
    train_dir_name="train",  # 自定义训练目录名（默认："train2017"）
    val_dir_name="val"       # 自定义验证目录名（默认："val2017"）
)
```

#### 完整转换过程（COCO到YOLO）

```python
from coco2yolo_kaggle import convert_coco_dataset

# 完整处理过程，使用自定义目录名
dataset_path = convert_coco_dataset(
    json_dir="/kaggle/input/coco-2017-dataset/coco2017/annotations",
    output_dir="/kaggle/working/coco_yolo",
    final_dest="/kaggle/tmp/COCO2017",
    use_segments=True,
    cls91to80=True,  # 将91个COCO类别映射到80个类别（默认：True）
    max_workers=8,
    copy_files=True,  # 设置为False可跳过复制文件
    train_dir_name="train",  # 自定义目标目录（默认："train2017"）
    val_dir_name="val",      # 自定义目标目录（默认："val2017"）
    src_train_dir_name="train2017",  # 源训练目录名
    src_val_dir_name="val2017"       # 源验证目录名
)
```

#### 仅转换标签（YOLO到COCO）

```python
from coco2yolo_kaggle import convert_yolo_labels

# 使用YAML文件提供类别信息
result = convert_yolo_labels(
    yolo_dataset_path="/kaggle/input/yolo-dataset",
    output_dir="/kaggle/working/yolo_coco",
    yaml_path="/kaggle/input/yolo-dataset/data.yaml"
)

# 或者直接提供类别名称
result = convert_yolo_labels(
    yolo_dataset_path="/kaggle/input/yolo-dataset",
    output_dir="/kaggle/working/yolo_coco",
    class_names=["person", "car", "bicycle"]
)
```

#### 完整转换过程（YOLO到COCO）

```python
from coco2yolo_kaggle import convert_yolo_dataset

# 完整处理过程
result = convert_yolo_dataset(
    yolo_dataset_path="/kaggle/input/yolo-dataset",
    output_dir="/kaggle/working/yolo_coco",
    final_dest="/kaggle/tmp/COCO",
    yaml_path="/kaggle/input/yolo-dataset/data.yaml",
    copy_images=True,
    max_workers=8
)
```

### 命令行使用

#### COCO到YOLO转换

##### 仅转换标签（默认模式）

```bash
coco2yolo-kaggle coco2yolo --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo
```

默认情况下，这将把91个COCO类别映射到80个类别。如果要保留原始的91个类别，请使用：

```bash
coco2yolo-kaggle coco2yolo --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --no-cls91to80
```

##### 仅复制文件（使用自定义目录名）

```bash
coco2yolo-kaggle coco2yolo --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val
```

##### 完整过程（使用自定义目录名）

```bash
coco2yolo-kaggle coco2yolo --mode=all --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val
```

#### YOLO到COCO转换

##### 使用类别名称转换标签

```bash
coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --class-names="person,car,bicycle"
```

##### 使用YAML文件转换标签

```bash
coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --yaml-path=/kaggle/input/yolo-dataset/data.yaml
```

##### 完整过程（包含复制图像）

```bash
coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --final-dest=/kaggle/tmp/COCO --yaml-path=/kaggle/input/yolo-dataset/data.yaml --copy-images
```

## 类别映射和自定义目录结构

### 类别映射

- 默认情况下，该工具将原始的91个COCO类别映射到标准的80个类别（`cls91to80=True`）
- 要保留原始的91个类别，请使用`--no-cls91to80`参数

### 目录名称自定义
您可以使用以下参数自定义目录名称：

- `--train-dir-name`：设置目标训练目录名称（默认："train2017"）
- `--val-dir-name`：设置目标验证目录名称（默认："val2017"）
- `--src-train-dir`：源训练目录名称（默认："train2017"）
- `--src-val-dir`：源验证目录名称（默认："val2017"）


## Kaggle示例代码

### COCO到YOLO示例

```python
# 1. 安装软件包
!pip install coco2yolo-kaggle

# 2. 仅转换标签（带类别映射）
!coco2yolo-kaggle coco2yolo --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo

# 3. 使用自定义目录名称复制数据集文件
!coco2yolo-kaggle coco2yolo --mode=copy --json-dir=/kaggle/input/coco-2017-dataset/coco2017/annotations --output-dir=/kaggle/working/coco_yolo --final-dest=/kaggle/tmp/COCO2017 --train-dir-name=train --val-dir-name=val

# 4. 训练YOLOv8模型
!pip install ultralytics
from ultralytics import YOLO

# 创建带有自定义目录结构的数据集配置文件
%%writefile coco.yaml
path: /kaggle/tmp/COCO2017
train: images/train  # 使用自定义目录名
val: images/val      # 使用自定义目录名
nc: 80
names: ['person', 'bicycle', 'car', ... ] # 完整的80个类别名称

# 训练模型
model = YOLO('yolov8n.pt')  # 使用nano模型
results = model.train(data='coco.yaml', epochs=3, imgsz=640)
```

### YOLO到COCO示例

```python
# 1. 安装软件包
!pip install coco2yolo-kaggle

# 2. 使用YAML文件将YOLO标签转换为COCO格式
!coco2yolo-kaggle yolo2coco --yolo-dataset-path=/kaggle/input/yolo-dataset --output-dir=/kaggle/working/yolo_coco --yaml-path=/kaggle/input/yolo-dataset/data.yaml

# 3. 使用Python API获得完全控制
from coco2yolo_kaggle import convert_yolo_dataset

# 使用更多选项将YOLO数据集转换为COCO
result = convert_yolo_dataset(
    yolo_dataset_path='/kaggle/input/yolo-dataset',
    output_dir='/kaggle/working/yolo_coco',
    final_dest='/kaggle/tmp/COCO',
    class_names=['person', 'car', 'bicycle'],  # 明确指定类别名称
    split_names=('train', 'val'),              # 要处理的数据集拆分
    image_ext='jpg',                           # 图像文件扩展名
    copy_images=True,                          # 将图像复制到最终目标
    max_workers=8                              # 并行处理线程数
)

print(f"生成的标注文件: {result}")
```

## 致谢

该工具基于[ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)，感谢原作者的贡献。

