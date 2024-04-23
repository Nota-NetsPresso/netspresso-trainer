# Data

NetsPresso Trainer supports learning functions for various vision tasks with your custom data. 
In addition to data stored in a local repository, it also supports learning with data accessible through APIs such as [Hugging Face datasets](https://huggingface.co/datasets). 
Currently, the dataset formats supported by NetsPresso Trainer are fixed in a specific form, but we plan to expand to more dataset formats such as COCO format in the future.  

On this page, we will guide you on the data format you need to learn with your custom data and how to learn using Hugging Face datasets. 

## Local custom datasets

### Supporting image formats

For image data, various extension images are supported, but we recommend one of `.jpg`, `.jpeg`, `.png`, and `.bmp`. In this case, label data used in semantic segmentation must be saved as `.png` to prevent data loss and utilize image header information. 
The following sections introduce how to organize data for each task. 

### Common configuration

Regardless of the task, dataset directory should be organized as follows:

- **Train**: This directory should contain all the training images and corresponding label files.
- **Validation**: This directory should house validation images and their corresponding labels, used to tune the hyperparameters.
- **Test**: This directory should include test images and labels for final model evaluation.

This structure should be reflected in your configuration file under the respective paths.

| Field <img width=200/> | Description |
|---|---|
| `data.name` | (str) The name of dataset. |
| `data.task` | (str) `classification` for image classification, `segmentation` for semantic segmentation, and `detection` for object detection. |
| `data.format` | **`local`** as an identifier of dataset format. |
| `data.path.root` | (str) Root directory of dataset. |
| `data.path.train.image` | (str) The directory for training images. Should be **relative** path to root directory. | 
| `data.path.valid.image` | (str) The directory for validation images. Should be **relative** path to root directory. | 
| `data.path.test.image` | (str) The directory for test images. Should be **relative** path to root directory. | 

### Image classification

To train an image classification model using NetsPresso Trainer, **users must organize their data according to a specified format.**

- train images must be in same directory.
- validation images must be in same directory.
- labels for images are given by csv file. The csv file contains image file name and correspoinding class label.

| Field <img width=200/> | Description |
|---|---|
| `data.id_mapping` | (list) Class list for each class index.  |
| `data.path.train.label` | (str) For classificaiton, label field must be path of `.csv` file. This should be **relative** path to root directory. | 
| `data.path.valid.label` | (str) For classificaiton, label field must be path of `.csv` file. This should be **relative** path to root directory. | 
| `data.path.test.label` | (str) For classificaiton, label field must be path of `.csv` file. This should be **relative** path to root directory. | 

<details>
  <summary>Data hierarchy example - ImageNet1K</summary>
  ```text
  data/imagenet1k
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
</details>

<details>
  <summary>Label csv example - ImageNet1K</summary>
  ```text
  | image_id             | class    |
  |----------------------|----------|
  | n03792972_3671.JPEG  | 728      |
  | n04357314_4256.JPEG  | 810      |
  | n02965783_127.JPEG   | 576      |
  | n04465501_16825.JPEG | 289      |
  | n09246464_5059.JPEG  | 359      |
  | ... | ... |
  ```
</details>

<details>
  <summary>Data configuration example - ImageNet1K</summary>
  ```yaml
  data:
    name: imagenet1k
    task: classification
    format: local # local, huggingface
    path:
      root: path_to/IMAGENET1K # dataset root
      train:
        image: images/train # directory for training images
        label: labels/imagenet_train.csv  # label for training images
      valid:
        image: images/valid  # directory for valid images
        label: labels/imagenet_valid.csv  # label for valid images
      test:
        image: ~  # directory for test images
        label: ~  # label for test images
    id_mapping: ["kit fox", "English setter", "Siberian husky", "Australian terrier", ...]
  ```
</details>

### Semantic segmentation

To train a semantic segmentation model using NetsPresso Trainer, **the data must be in the following formats**: 

- For each training image, there must be a label file (image) indicating the original image and the class index of each pixel of the image.
- Users must create an image and label directory under the root directory and put the corresponding files in each directory.
- In this case, training data and validation data can be distinguished in different directories. For example, training data can be placed in train/image, train/label directories, and validation data can be placed in valid/image, valid/label directories.
- Users must know the class name corresponding to each pixel value (RGB or L (grayscale) format) in the label file.

| Field <img width=200/> | Description |
|---|---|
| `data.label_image_mode` | (str) Image mode to convert the label. Should be one of `RGB`, `L`, and `P`. This field is not case-sensitive. |
| `data.path.train.label` | (str) For segmentation, label field must be path of label directory. This should be **relative** path to root directory. | 
| `data.path.valid.label` | (str) For segmentation, label field must be path of label directory. This should be **relative** path to root directory. | 
| `data.path.test.label` | (str) For segmentation, label field must be path of label directory. This should be **relative** path to root directory. |
| `data.id_mapping` | (dict, list) Key-value pair between label value (`RGB`, `L`, or `P`) and class name. Should be a dict of {**label_value: classname**} or a list of class names whose indices are same with the label value (image_mode: `L` or `P`). |
| `data.palette` | (dict) Color mapping for visualization. If `none`, automatically select the color for each class.  |


<details>
  <summary>Data hierarchy example - PascalVOC 2012</summary>
  ```text
  data/voc2012_seg
  ├── images
  │   ├── train
  │   │   ├── 2007_000032.jpg
  │   │   ├── 2007_000039.jpg
  │   │   ├── 2007_000063.jpg
  │   │   └── ...
  │   └── valid
  │       ├── 2007_000033.jpg
  │       ├── 2007_000042.jpg
  │       ├── 2007_000061.jpg
  │       └── ...
  └── labels
      ├── train
      │   ├── 2007_000032.png
      │   ├── 2007_000039.png
      │   ├── 2007_000063.png
      │   └── ...
      └── valid
          ├── 2007_000033.png
          ├── 2007_000042.png
          ├── 2007_000061.png
          └── ...
  ```
</details>

<details>
  <summary>Data configuration example - PascalVOC 2012</summary>
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
        image: image/valid
        label: mask/valid
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
</details>

### Object detection

To train an object detection model using NetsPresso Trainer, **the data must be in the following formats**: 

- For object detection model training, each training image must have a corresponding `.txt` file indicating the original image and the bounding box and class index corresponding to each bounding box of the image.
- The format of the bounding box follows the YOLO dataset format `[x_center, y_center, width, height]` (normalized).
- Each `.txt` file must contain one line for each bounding box.
- In this case, training data and validation data can be distinguished in different directories. For example, training data can be placed in train/image, train/label directories, and validation data can be placed in valid/image, valid/label directories.
- Users must know the class name corresponding to each class index in the label file.

| Field <img width=200/> | Description |
|---|---|
| `data.path.train.label` | (str) For detection, label field must be path of label directory. This should be **relative** path to root directory. | 
| `data.path.valid.label` | (str) For detection, label field must be path of label directory. This should be **relative** path to root directory. | 
| `data.path.test.label` | (str) For detection, label field must be path of label directory. This should be **relative** path to root directory. |
| `data.id_mapping` | (list) Class list for each class index. |
| `data.palette` | (dict) Color mapping for visualization. If `none`, automatically select the color for each class. |

<details>
  <summary>Data hierarchy example - COCO 2017</summary>
  ```text
  data/coco2017
  ├── images
  │   ├── train
  │   │   ├── 000000000009.jpg
  │   │   ├── 000000000025.jpg
  │   │   ├── 000000000030.jpg
  │   │   └── ...
  │   └── valid
  │       ├── 000000000139.jpg
  │       ├── 000000000285.jpg
  │       ├── 000000000632.jpg
  │       └── ...
  └── labels
      ├── train
      │   ├── 000000000009.txt
      │   ├── 000000000025.txt
      │   ├── 000000000030.txt
      │   └── ...
      └── valid
          ├── 000000000139.txt
          ├── 000000000285.txt
          ├── 000000000632.txt
          └── ...
  ```
</details>

<details>
  <summary>Label txt example - COCO 2017</summary>
  ```text
  58 0.389578125 0.4161032863849765 0.038593749999999996 0.16314553990610328
  62 0.127640625 0.5051525821596244 0.23331249999999998 0.22269953051643193
  62 0.9341953125 0.583462441314554 0.127109375 0.18481220657276995
  56 0.60465625 0.6325469483568076 0.0875 0.24138497652582158
  56 0.5025078125 0.6273239436619719 0.096609375 0.2311737089201878
  56 0.6691953125 0.6189906103286384 0.047140625000000005 0.19098591549295774
  56 0.512796875 0.5282511737089202 0.03371875 0.02720657276995305
  0 0.6864453125 0.5319600938967136 0.082890625 0.3239671361502347
  0 0.612484375 0.4461971830985916 0.023625 0.08389671361502347
  68 0.811859375 0.5017253521126761 0.02303125 0.037488262910798126
  72 0.7863203125 0.5363732394366197 0.031703125 0.2542488262910798
  73 0.9561562499999999 0.7717018779342724 0.02240625 0.10730046948356808
  73 0.96825 0.7780751173708921 0.020125 0.10901408450704225
  74 0.7105546875 0.31 0.021828125 0.05136150234741784
  75 0.8865624999999999 0.8316079812206573 0.0573125 0.2104929577464789
  75 0.5569453125 0.5167018779342724 0.017765625 0.05293427230046949
  56 0.6516640625 0.5288262910798122 0.015046875000000001 0.029389671361502348
  75 0.388046875 0.4784154929577465 0.022218750000000002 0.04138497652582159
  75 0.5338359375 0.48794600938967136 0.015203125000000001 0.039272300469483566
  60 0.599984375 0.6471478873239437 0.19618750000000001 0.20875586854460096

  ```
</details>

<details>
  <summary>Custom object detection dataset example - COCO 2017</summary>
  ```yaml
  data:
    name: coco2017
    task: detection
    format: local # local, huggingface
    path:
      root: ./data/coco2017 # dataset root
      train:
        image: images/train # directory for training images
        label: labels/train # directory for training labels
      valid:
        image: images/valid # directory for valid images
        label: labels/valid # directory for valid labels
      test:
        image: ~
        label: ~
      pattern:
        image: ~
        label: ~
    id_mapping: ['person', 'bicycle', 'car', ...]
    pallete: ~
  ```
</details>

## Hugging Face datasets

NetsPresso Trainer is striving to support various dataset hubs and platforms. 
As part of that effort and first step, NetsPresso Trainer can be used with data in [Hugging Face datasets](https://huggingface.co/datasets). 

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

<details>
  <summary>Huggingface dataset example - beans</summary>
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
</details>