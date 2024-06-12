<div align="center">
    <img src="./assets/netspresso_trainer_header_tmp.png" width="800"/>
</div>
</br>

<center style="white-space: pre-line">
Start training models (including ViTs) with <b>NetsPresso Trainer</b>,
compress and deploy your model with <b>NetsPresso</b>!
</center>
</br>

<div align="center">
<p align="center">
  <a href="https://py.netspresso.ai/">Website</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="https://github.com/Nota-NetsPresso/netspresso-trainer/issues">Issues</a> •
  <a href="https://nota-netspresso.github.io/netspresso-trainer">Docs</a>
</p>
</div>

_____

## Table of contents

<!-- toc -->

- [Installation](#installation)
- [Getting started](#getting-started)

<!-- tocstop -->

## Installation (Stable)

### Prerequisites

- Python `3.8` | `3.9` | `3.10`
- PyTorch `1.13.0` (recommended) (compatible with: `1.11.x` - `1.13.x`)

### Install with pypi

```bash
pip install netspresso_trainer
```

### Install with GitHub

```bash
pip install git+https://github.com/Nota-NetsPresso/netspresso-trainer.git@master
```

To install with editable mode,

```bash
git clone -b master https://github.com/Nota-NetsPresso/netspresso-trainer.git
pip install -e netspresso-trainer
```

### Set-up with docker

Please clone this repository and refer to [`Dockerfile`](./Dockerfile) and [`docker-compose-example.yml`](./docker-compose-example.yml).  
For docker users, we provide more detailed guide in our [Docs](https://nota-netspresso.github.io/netspresso-trainer).

## Getting started

Write your training script in `train.py` like:

```python
from netspresso_trainer import train_cli

if __name__ == '__main__':
    logging_dir = train_cli()
    print(f"Training results are saved at: {logging_dir}")
```

Then, train your model with your own configuraiton:

```bash
python train.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --training config/training/classification.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

Or you can start NetsPresso Trainer by just executing console script which has same feature.

```bash
netspresso-train\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --training config/training/classification.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

Please refer to [`scripts/example_train.sh`](./scripts/example_train.sh).

NetsPresso Trainer is compatible with [NetsPresso](https://netspresso.ai/) service. We provide NetsPresso Trainer tutorial that contains whole procedure from model train to model compression and benchmark. Please refer to our [colab tutorial](https://colab.research.google.com/drive/1RBKMCPEa4x-4X31zqzTS8WgQI9TQt3e-?usp=sharing).

## Dataset preparation (Local)

NetsPresso Trainer is designed to accommodate a variety of tasks, each requiring different dataset formats. You can find the specific dataset formats for each task in our [documentation](https://nota-netspresso.github.io/netspresso-trainer/components/data/).

If you are interested in utilizing open datasets, you can use them by following the [instructions](https://nota-netspresso.github.io/netspresso-trainer/getting_started/dataset_preparation/local/#open-datasets).

### Image classification

- [CIFAR100](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/cifar100.py)
- [ImageNet1K](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/imagenet1k.py)

### Semantic segmentation

- [ADE20K](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/ade20k.py)
- [Cityscapes](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/cityscapes.py)
- [PascalVOC 2012](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/voc2012_seg.py)

### Object detection

- [COCO 2017](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/coco2017.py)

### Pose estimation

- [WFLW](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/tools/open_dataset_tool/wflw.py)

## Dataset preparation (Huggingface)

NetsPresso Trainer is also compatible with huggingface dataset. To use datasets of huggingface, please check [instructions in our documentations](https://nota-netspresso.github.io/netspresso-trainer/getting_started/dataset_preparation/huggingface/). This enables to utilize a wide range of pre-built datasets which are beneficial for various training scenarios.

## Pretrained weights

Please refer to our [official documentation](https://nota-netspresso.github.io/netspresso-trainer/) for pretrained weights supported by NetsPresso Trainer.

## Tensorboard

We provide basic tensorboard to track your training status. Run the tensorboard with the following command: 

```bash
tensorboard --logdir ./outputs --port 50001 --bind_all
```

where `PORT` for tensorboard is 50001.  
Note that the default directory of saving result will be `./outputs` directory.