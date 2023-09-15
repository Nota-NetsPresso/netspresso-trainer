## Overview

NetsPresso Trainer supports learning functions for various vision tasks with your custom data. 
In addition to data stored in a local repository, it also supports learning with data accessible through APIs such as [Hugging Face datasets](https://huggingface.co/datasets). 
Currently, the dataset formats supported by NetsPresso Trainer are fixed in a specific form, but we plan to expand to more dataset formats such as COCO format in the future.  

On this page, we will guide you on the data format you need to learn with your custom data and how to learn using Hugging Face datasets. 
## Supporting format

For image data, various extension images are supported, but we recommend one of `.jpg`, `.jpeg`, `.png`, and `.bmp`. In this case, label data used in semantic segmentation must be saved as `.png` to prevent data loss and utilize image header information. 
The following sections introduce how to organize data for each task. 

### Image classification

To train an image classification model using NetsPresso Trainer, **the data must be in following formats**: 

- There must be a directory for each class to be distinguished by the classification model.
- Each class directory must contain all the images corresponding to that class.
- Collect directories containing images for each class under the root directory.
- Users must know in advance which class name each class directory name corresponds to.
 
The example data directory structure for this is as follows:

```
# TODO
```

An example yaml configuration for this is as follows:

```yaml
data:
  name: food_pic
  task: classification
  format: local # local, huggingface
  path:
    root: ./data/my_food_pics # dataset root
    train:
      image: train # directory for training images
      label: ~  # label for training images
    valid:
      image: val  # directory for valid images
      label: ~  # label for valid images
    test:
      image: ~  # directory for test images
      label: ~  # label for test images
  id_mapping:  # Dict[directory_name, class_name]. If None, set the directory name same with class name
    directory_1: curry
    directory_2: ramen
    directory_3: rice
    directory_4: sushi

```

### Semantic segmentation

To train a semantic segmentation model using NetsPresso Trainer, **the data must be in following formats**: 

- For each training image, there must be a label file (image) indicating the original image and the class index of each pixel of the image.
- Users must create an image and label directory under the root directory and put the corresponding files in each directory.
- In this case, training data and validation data can be distinguished in different directories. For example, training data can be placed in train/image, train/label directories, and validation data can be placed in valid/image, valid/label directories.
- Users must know the class name corresponding to each pixel value (RGB or L (grayscale) format) in the label file.

The example data directory structure for this is as follows:

```
# TODO

```

An example yaml configuration for this is as follows:

*FIXME* https://github.com/Nota-NetsPresso/netspresso-trainer/issues/150

```yaml

```

### Object detection

To train an object detection model using NetsPresso Trainer, **the data must be in following formats**: 

- For object model training, there must be a `.txt` file for each training image indicating the original image and the bounding box and class index corresponding to each bounding box of the image.
- The format of the bounding box follows the YOLO dataset format ([x_center, y_center, width, height], normalized).
- Each `.txt` file must contain one line for each bounding box.
- In this case, training data and validation data can be distinguished in different directories. For example, training data can be placed in train/image, train/label directories, and validation data can be placed in valid/image, valid/label directories.
- Users must know the class name corresponding to each class index in the label file.

The example data directory structure for this is as follows: 

```
# TODO

```

An example yaml configuration for this is as follows: 

```yaml
# This example dataset is downloaded from <https://www.kaggle.com/code/valentynsichkar/traffic-signs-detection-by-yolo-v3-opencv-keras/input>
data:
  name: traffic_sign_yolo
  task: detection
  format: local # local, huggingface
  path:
    root: ../../data/traffic-sign # dataset root
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

- `data.name` : 
- `data.task` : 
- `data.format` : 
- `data.path.root` : 
- `data.path.train.image` : 
- `data.path.train.label` : 
- `data.path.valid.image` : 
- `data.path.valid.label` : 
- `data.path.test.image` : 
- `data.path.test.label` :

#### Classification

- `data.path.id_mapping` :

#### Segmentation

- `data.path.id_mapping` :
- `data.path.palette` :

#### Detection

- `data.path.id_mapping` :
- `data.path.palette` :



### Hugging Face datasets

- `data.name` :
- `data.task` :
- `data.format` :
- `data.metadata.custom_cache_dir` :
- `data.metadata.repo` :
- `data.metadata.subset` :
- `data.metadata.features.image` :
- `data.metadata.features.label` :