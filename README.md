# NetsPresso trainer all-in-one

This repository aims to support training PyTorch models which are compatible with [NetsPresso](https://netspresso.ai/),
model compressor platform from [Nota AI](https://www.nota.ai/).

## Installation with docker

### Docker with docker-compose

For the latest information, please check [`docker-compose.yml`](./docker-compose.yml)

```bash
# run command
docker compose run --service-ports --name trainer-allinone-dev modelsearch-trainer-allinone bash
```

### Docker image build

If you run with `docker run` command, follow the image build and run command in the below:

```bash
# build an image
export $(cat .env | xargs) && \
docker build -t modelsearch-trainer-allinone:$TAG .
```

```bash
# docker run command
export $(cat .env | xargs) && \
docker run -it --ipc=host\
  --gpus='"device=0,1,2,3"'\
  -v /PATH/TO/modelsearch-trainer-allinone:/workspace\
  -v /PATH/TO/DATA:/DATA\
  -v /PATH/TO/CHECKPOINT:/CHECKPOINT\
  -p 50001:50001\
  --name trainer-allinone-dev modelsearch-trainer-allinone:$TAG
```

## Example training

This code provides some example scripts and snippets to help you understand about the functionalities.

### Training example model

For classification and segmentation, see [`train_classification.sh`](./train_classification.sh) [`train_segmentation.sh`](./train_segmentation.sh) for each.  
Each shell sciprt contains two commands: (1) multi-gpu training and (2) single-gpu training.
A default option is using **single-gpu**, but you can edit the script if you needed.

> :warning: `2023.06.21` Work in progress for detection (It won't work for now)

### Tensorboard

Please refer to [`run_tensorboard.sh`](./run_tensorboard.sh).
To execute on background, you may run this script with `tmux`, `screen`, or `bg`.