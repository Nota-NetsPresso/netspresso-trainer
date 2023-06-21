FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN apt-get update && apt-get install -y git vim curl zip unzip wget htop ncdu tmux screen libgl1-mesa-glx libglib2.0-0 

RUN mkdir -p /workspace
WORKDIR /workspace

COPY requirements.txt .

RUN python -m pip install --upgrade pip 
RUN python -m pip install --no-cache-dir -r requirements.txt
