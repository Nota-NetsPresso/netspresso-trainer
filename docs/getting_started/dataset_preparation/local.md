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

- For image classification, you may need csv format label files.
- For semantic segmentation and object detection, organize your label files (could be masks or box annotations) in corresponding folders.

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

Define the paths to your datasets in the configuration file to tell NetsPresso Trainer where to find the data. Finally, you can complete data configuration by adding some metadata like `id_mapping`. Here is example for classification:

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

If you are interested in using open datasets, follow the instructions below to seamlessly integrate them into the local custom datasets format.

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

#### ADE20K

Run `ade20k.py` python file with your dataset directory as an augument.

ADE20K dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/ade20k.yaml).

```bash
python ./tools/open_dataset_tool/ade20k.py --dir ./data
```

#### Cityscapes

Cityscapes dataset cannot be automatically downloaded. You should download dataset from [Cityscapes](https://www.cityscapes-dataset.com/) website, and place downloaded files into `./data/download`.

And, run `cityscapes.py` python file with your dataset directorty and downloaded files path as arguments. After executing scripts, you can use [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/cityscapes.yaml).

```bash
python --dir ./data --images .data/download/leftImg8bit_trainvaltest.zip --labels .data/download/gtFine_trainvaltest.zip
```

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

#### Objects365
Run `objects365.py` python file with your dataset directory as an argument.

Objects365 dataset will be automatically downloaded to `./data/download/objects365`. After executing scripts, you can use [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/objects365.yaml). As the dataset is quite large, It is recommened to use multiprocess when you download it (e.g., `--num_process 4`).

```bash
python ./tools/open_dataset_tool/objects365.py --dir ./data --num_process 4
```

#### PascalVOC 2012
 
Run `voc2012_det.py` python file with your dataset directory as an argument.

PascalVOC 2012 dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/voc2012_det.yaml).

```bash
python ./tools/open_dataset_tool/voc2012_det.py --dir ./data
```

### Pose estimation

#### WFLW

Run `wflw.py` python file with your dataset directory as an argument.

WFLW dataset will be automatically downloaded to `./data/download`. After executing scripts, you can use  [pre-defined configuration](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/config/data/local/wflw.yaml).

```bash
python ./tools/open_dataset_tool/wflw.py --dir ./data
```

## Run NetsPresso Trainer

Now you can run NetsPresso Trainer with your local dataset!

```bash
python train.py --data your_huggingface_dataset_yaml_path.yaml ...
```