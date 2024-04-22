# Data preparation (Local)

## Local custom datasets

If your dataset is ready in local storage, you can use them by following the instructions.

### Organize dataset

Create separate directories for images/labels and train/valid/test.

```text
/my_dataset
├── images
│   ├── train
│   ├── valid
│   └── test
└── labels
```

Place your images on proper path.

```text
/my_dataset
├── images
│   ├── train
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── valid
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── test
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── labels
```

Set labels on proper path.

- For image classification, you may need image files and a corresponding label file (usually in csv format).
- For semantic segmentation and object detection, organize your images and label files (could be masks or box annotations) in corresponding folders.

```text
/my_dataset
├── images
│   ├── train
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── valid
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── test
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── labels
    └── train directory or file ...
    ├── valid directory or file ...
    └── test directory or file ...
```

*If you just run training, test split may not needed.*

*If you just run evaluation or inference, train and valid split may not needed.*

### Set configuration file

Define the paths to your datasets in the configuration file to tell NetsPresso Trainer where to find the data. Here is example for classification:

```yaml
data:
  name: my_custom_dataset
  task: classification # This could be other task
  format: local
  path:
    root: ./my_dataset
    train:
      image: train/images
      label: train/labels.csv
    valid:
      image: valid/images
      label: valid/labels.csv
    test:
      image: test/images
      label: test/labels.csv
  id_mapping: [cat, dog, elephant]
```

For detailed definition of data configuration, please refer to [components/data](../../components/data.md)

## Open datasets

If you are interested in utilizing open datasets, you can use them by following the instructions. These instructions automatically set dataset as local custom datasets format.

### Image classification

#### CIFAR100

Run `cifar100.py` python file with your dataset directory as an argument.

CIFAR100 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/cifar100.yaml).

```bash
python ./tools/open_dataset_tool/cifar100.py --dir ./data
```

#### ImageNet1K

ImageNet1K dataset cannot be automatically downloaded. You should download dataset from [ImageNet](https://www.image-net.org/) website, and place downloaded files into `./data/download`.

And, run `imagenet1k.py` python file with your dataset directorty and downloaded files path as arguments. After executing scripts, you can use [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/imagenet1k.yaml).

*(`imagenet1k.py` needs scipy library which is in [requirements-optional.txt](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/requirements-optional.txt))*

```bash
python ./tools/open_dataset_tool/imagenet1k.py --dir ./data --train-images ./data/download/ILSVRC2012_img_train.tar --valid-images ./data/download/ILSVRC2012_img_val.tar --devkit ./data/download/ILSVRC2012_devkit_t12.tar.gz
```

### Semantic segmentation

#### PascalVOC 2012
 
Run `voc2012_seg.py` python file with your dataset directory as an argument.

PascalVOC 2012 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/voc12.yaml).

```bash
python ./tools/open_dataset_tool/voc2012_seg.py --dir ./data
```

### Object detection

#### COCO 2017

Run `coco2017.py` python file with your dataset directory as an argument.

COCO 2017 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/coco2017.yaml).

```bash
python ./tools/open_dataset_tool/coco2017.py --dir ./data
```

## Run NetsPresso Trainer

Now you can run NetsPresso Trainer with your local dataset!

```bash
python train.py --data your_huggingface_dataset_yaml_path.yaml ...
```