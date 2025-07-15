FROM python:3.10

ARG TORCH_VERSION="2.0.1"
ARG TORCHVISION_VERSION="0.15.2"

RUN apt-get update && \
    apt-get install -y \
    git \ 
    vim \
    curl \
    zip \ 
    unzip \ 
    wget \
    htop \
    ncdu \
    tmux \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
# RUN python -m pip install --no-cache-dir torch==${TORCH_VERSION}+cu116 torchvision==${TORCHVISION_VERSION}+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} && rm -rf /root/.cache/pip

RUN mkdir -p /home/appuser/netspresso-trainer
WORKDIR /home/appuser/netspresso-trainer

COPY . /home/appuser/netspresso-trainer

RUN pip install -r requirements.txt && rm -rf /root/.cache/pip
RUN python3 -m pip install -e .
