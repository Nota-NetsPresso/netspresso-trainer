# Data

NetsPresso Trainer supports learning functions for various vision tasks with your custom data. 
In addition to data stored in a local repository, it also supports learning with data accessible through APIs such as [Hugging Face datasets](https://huggingface.co/datasets). 
Currently, the dataset formats supported by NetsPresso Trainer are fixed in a specific form, but we plan to expand to more dataset formats such as COCO format in the future.  

On this page, we will guide you on the data format you need to learn with your custom data and how to learn using Hugging Face datasets. 
## Supporting format

For image data, various extension images are supported, but we recommend one of `.jpg`, `.jpeg`, `.png`, and `.bmp`. In this case, label data used in semantic segmentation must be saved as `.png` to prevent data loss and utilize image header information. 
The following sections introduce how to organize data for each task. 

## Training with your custom datasets

### Image classification

To train an image classification model using NetsPresso Trainer, **users must organize their data according to a specified format.**

- train images must be in same directory.
- validation images must be in same directory.
- labels for images are given by csv file. The csv file contains image file name and correspoinding class label.

The example data directory structure for this is as follows. The following examples use the [ImageNet1K](https://image-net.org/) dataset.:

```
IMAGENET1K
├── images
│   ├── train
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   └── ...
│   └── valid
│       ├── ILSVRC2012_val_00000001.JPEG
│       ├── ILSVRC2012_val_00000002.JPEG
│       ├── ILSVRC2012_val_00000003.JPEG
│       └── ...
└── labels
    ├── imagenet_train.csv
    └── imagenet_valid.csv
```

An example yaml configuration for this is as follows:

```yaml
data:
  name: imagenet1k
  task: classification
  format: local # local, huggingface
  path:
    root: path_to/IMAGENET1K # dataset root
    train:
      image: train # directory for training images
      label: imagenet_train.csv  # label for training images
    valid:
      image: val  # directory for valid images
      label: imagenet_valid.csv  # label for valid images
    test:
      image: ~  # directory for test images
      label: ~  # label for test images
  id_mapping: ["kit fox", "English setter", "Siberian husky", "Australian terrier", ...]
```

An example csv label for this is as follows:

| image_id             | class    |
|----------------------|----------|
| n03792972_3671.JPEG  | 728      |
| n04357314_4256.JPEG  | 810      |
| n02965783_127.JPEG   | 576      |
| n04465501_16825.JPEG | 289      |
| n09246464_5059.JPEG  | 359      |
| ... | ... |

### Semantic segmentation

To train a semantic segmentation model using NetsPresso Trainer, **the data must be in the following formats**: 

- For each training image, there must be a label file (image) indicating the original image and the class index of each pixel of the image.
- Users must create an image and label directory under the root directory and put the corresponding files in each directory.
- In this case, training data and validation data can be distinguished in different directories. For example, training data can be placed in train/image, train/label directories, and validation data can be placed in valid/image, valid/label directories.
- Users must know the class name corresponding to each pixel value (RGB or L (grayscale) format) in the label file.

The example data directory structure for this is as follows:

```
VOC12Dataset
├── image
│   ├── train
│   │   ├── 2007_000032.jpg
│   │   ├── 2007_000039.jpg
│   │   ├── 2007_000063.jpg
│   │   └── ...
│   └── val
│       ├── 2007_000033.jpg
│       ├── 2007_000042.jpg
│       ├── 2007_000061.jpg
│       └── ...
└── mask
    ├── train
    │   ├── 2007_000032.png
    │   ├── 2007_000039.png
    │   ├── 2007_000063.png
    │   └── ...
    └── val
        ├── 2007_000033.png
        ├── 2007_000042.png
        ├── 2007_000061.png
        └── ...
```

An example yaml configuration for this is as follows:

```yaml
data:
  name: voc2012
  task: segmentation
  format: local
  path:
    root: path_to/VOC12Dataset
    train:
      image: image/train
      label: mask/train
    valid:
      image: image/val
      label: mask/val
    test:
      image: ~  # directory for test images
      label: ~  # directory for test labels
    pattern:
      image: ~
      label: ~
  label_image_mode: RGB
  id_mapping:
    (0, 0, 0): background
    (128, 0, 0): aeroplane
    (0, 128, 0): bicycle
    (128, 128, 0): bird
    (0, 0, 128): boat
    (128, 0, 128): bottle
    (0, 128, 128): bus
    (128, 128, 128): car
    (64, 0, 0): cat
    (192, 0, 0): chair
    (64, 128, 0): cow
    (192, 128, 0): diningtable
    (64, 0, 128): dog
    (192, 0, 128): horse
    (64, 128, 128): motorbike
    (192, 128, 128): person
    (0, 64, 0): pottedplant
    (128, 64, 0): sheep
    (0, 192, 0): sofa
    (128, 192, 0): train
    (0, 64, 128): tvmonitor
    (128, 64, 128): void
  pallete: ~
```

### Object detection

To train an object detection model using NetsPresso Trainer, **the data must be in the following formats**: 

- For object detection model training, each training image must have a corresponding `.txt` file indicating the original image and the bounding box and class index corresponding to each bounding box of the image.
- The format of the bounding box follows the YOLO dataset format ([x_center, y_center, width, height], normalized).
- Each `.txt` file must contain one line for each bounding box.
- In this case, training data and validation data can be distinguished in different directories. For example, training data can be placed in train/image, train/label directories, and validation data can be placed in valid/image, valid/label directories.
- Users must know the class name corresponding to each class index in the label file.

The example data directory structure for this is as follows: 

```
traffic-sign
├── images
│   ├── train
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   ├── 00003.jpg
│   │   └── ...
│   └── val
│       ├── 00002.jpg
│       ├── 00004.jpg
│       ├── 00015.jpg
│       └── ...
└── labels
    ├── train
    │   ├── 00000.txt
    │   ├── 00001.txt
    │   ├── 00003.txt
    │   └── ...
    └── val
        ├── 00002.txt
        ├── 00004.txt
        ├── 00015.txt
        └── ...
```

An example yaml configuration for this is as follows: 

```yaml
# This example dataset is downloaded from <https://www.kaggle.com/code/valentynsichkar/traffic-signs-detection-by-yolo-v3-opencv-keras/input>
data:
  name: traffic_sign_yolo
  task: detection
  format: local # local, huggingface
  path:
    root: path_to/traffic-sign # dataset root
    train:
      image: images/train # directory for training images
      label: labels/train # directory for training labels
    valid:
      image: images/val  # directory for valid images
      label: labels/val  # directory for valid labels
    test:
      image: ~  # directory for test images
      label: ~  # directory for test labels
    pattern:
      image: ~
      label: ~
  id_mapping: ['prohibitory', 'danger', 'mandatory', 'other']  # class names
  pallete: ~
```

An example txt label for this is as follows:

```
2 0.7378676470588236 0.5125 0.030147058823529412 0.055
2 0.3044117647058823 0.65375 0.041176470588235294 0.0725
3 0.736764705882353 0.453125 0.04264705882352941 0.06875
```

## Training with Hugging Face datasets

NetsPresso Trainer is striving to support various dataset hubs and platforms. 
As part of that effort and first step, NetsPresso Trainer can be used with data in [Hugging Face datasets](https://huggingface.co/datasets). 

An example configuration for Hugging Face datasets is as follows: 

```yaml
data:
  name: beans
  task: classification
  format: huggingface
  metadata:
    custom_cache_dir: ./data/huggingface 
    repo: beans
    subset: ~
    features:
      image: image
      label: labels

```

## Field list

### Local dataset

#### Common

| Field <img width=200/> | Description |
|---|---|
| `data.name` | (str) The name of dataset. |
| `data.task` | (str) `classification` for image classification, `segmentation` for semantic segmentation, and `detection` for object detection. |
| `data.format` | **`local`** as an identifier of dataset format. |
| `data.path.root` | (str) Root directory of dataset. |
| `data.path.train.image` | (str) The directory for training images. Should be **relative** path to root directory. | 
| `data.path.train.label` | (str) The directory for training labels. Should be **relative** path to root directory. | 
| `data.path.valid.image` | (str) The directory for validation images. Should be **relative** path to root directory. | 
| `data.path.valid.label` | (str) The directory for validation labels. Should be **relative** path to root directory. | 
| `data.path.test.image` | (str) The directory for test images. Should be **relative** path to root directory. | 
| `data.path.test.label` | (str) The directory for test labels. Should be **relative** path to root directory. | 

#### Classification

| Field <img width=200/> | Description |
|---|---|
| `data.id_mapping` | (dict) Key-value pair between directory name and class name. Should be a dict of {**dirname: classname**}. |

#### Segmentation

| Field <img width=200/> | Description |
|---|---|
| `data.label_image_mode` | (str) Image mode to convert the label. Should be one of `RGB`, `L`, and `P`. This field is not case-sensitive.
| `data.id_mapping` | (dict, list) Key-value pair between label value (`RGB`, `L`, or `P`) and class name. Should be a dict of {**label_value: classname**} or a list of class names whose indices are same with the label value (image_mode: `L` or `P`). |
| `data.palette` | (dict) Color mapping for visualization. If `none`, automatically select the color for each class.  |

#### Detection

| Field <img width=200/> | Description |
|---|---|
| `data.id_mapping` | (list) Class list for each class index. |
| `data.palette` | (dict) Color mapping for visualization. If `none`, automatically select the color for each class.  |


### Hugging Face datasets

| Field <img width=200/> | Description |
|---|---|
| `data.name` | (str) The name of dataset. |
| `data.task` | (str) `classification` for image classification, `segmentation` for semantic segmentation, and `detection` for object detection. |
| `data.format` | **`huggingface`** as an identifier of dataset format. |
| `data.metadata.custom_cache_dir` | (str) Cache directory to load and save dataset files from Hugging Face. |
| `data.metadata.repo` | (str) Repository name. (e.g. `competitions/aiornot` represents the dataset `huggingface.co/datasets/competitions/aiornot`.) | 
| `data.metadata.subset` | (str, optional) Subset name if the dataset contains multiple versions. | 
| `data.metadata.features.image` | (str) The key representing the image at the dataset header. | 
| `data.metadata.features.label` | (str) The key representing the label at the dataset header. | 
