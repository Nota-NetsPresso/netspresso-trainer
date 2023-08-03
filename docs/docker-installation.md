## Installation with docker

### Docker with docker-compose

For the latest information, please check [`docker-compose.yml`](./docker-compose.yml)

```bash
# run command
export TAG=v$(cat src/netspresso_trainer/VERSION) && \
docker compose run --service-ports --name netspresso-trainer-dev netspresso-trainer bash
```

### Docker image build

If you run with `docker run` command, follow the image build and run command in the below:

```bash
# build an image
docker build -t netspresso-trainer:v$(cat src/netspresso_trainer/VERSION) .
```

```bash
# docker run command
docker run -it --ipc=host\
  --gpus='"device=0,1,2,3"'\
  -v /PATH/TO/DATA:/DATA/PATH/IN/CONTAINER\
  -v /PATH/TO/CHECKPOINT:/CHECKPOINT/PATH/IN/CONTAINER\
  -p 50001:50001\
  -p 50002:50002\
  -p 50003:50003\
  --name netspresso-trainer-dev netspresso-trainer:v$(cat src/netspresso_trainer/VERSION)
```