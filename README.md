<div align="center">
    <img src="./assets/netspresso_trainer_header_tmp.png" width="800"/>
</div>
</br>

<center>
Start training models (including ViTs) with <b>NetsPresso Trainer</b>, compress and deploy your model with <b>NetsPresso</b>!
</center>
</br>

<div align="center">
<p align="center">
  <a href="https://py.netspresso.ai/">Website</a> •
  <a href="#getting-started">Getting Started</a>
</p>
</div>

_____


## Table of contents

<!-- toc -->

- [Getting started](#getting-started)
- [Installation](#installation)

<!-- tocstop -->

## Getting started

Write your training script in `train.py` like:

```python
from netspresso_trainer import set_arguments, train

args_parsed, args = set_arguments(is_graphmodule_training=False)
train(args_parsed, args, is_graphmodule_training=False)
```

Then, train your model with your own configuraiton:

```bash
python train.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/resnet.yaml\
  --model config/model/resnet.yaml\
  --training config/training/resnet.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

Please refer to [`example_train.sh`](./example_train.sh) and [`example_train_fx.sh`](./example_train_fx.sh).

## Installation

### Prerequisites

- Python `3.8` | `3.9` | `3.10`
- PyTorch `1.13.0` (recommended) (compatible with: `1.11.x` - `1.13.x`)

### Install with pypi (stable)

```bash
pip install netspresso_trainer
```

### Install with GitHub

```bash
pip install git+https://github.com:Nota-NetsPresso/netspresso-trainer.git@stable
```

To install with editable mode,

```bash
git clone https://github.com:Nota-NetsPresso/netspresso-trainer.git .
pip install -e netspresso-trainer
```

### Set-up with docker

Please clone this repository and refer to [`Dockerfile`](./Dockerfile) and [`docker-compose-example.yml`](./docker-compose-example.yml)