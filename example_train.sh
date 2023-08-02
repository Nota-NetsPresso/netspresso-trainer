# python train.py\
#   --data config/data/chess.yaml\
#   --augmentation config/augmentation/resnet.yaml\
#   --model config/model/resnet.yaml\
#   --training config/training/resnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml


# python train.py\
#   --data config/data/traffic-sign.yaml\
#   --augmentation config/augmentation/efficientformer-detection.yaml\
#   --model config/model/efficientformer-detection.yaml\
#   --training config/training/efficientformer.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml


#### Multi-GPU training
# Put the number of GPU(s) to use in training at `--nproc_per_node`
#### (END)

# python -m torch.distributed.launch
#   --nproc_per_node 2\
#   train.py\
#   --data config/data/chess.yaml\
#   --augmentation config/augmentation/resnet.yaml\
#   --model config/model/resnet.yaml\
#   --training config/training/resnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml


# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/pidnet.yaml\
#   --model config/model/pidnet.yaml\
#   --training config/training/pidnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml


# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/coco_yolo.yaml\
#   --augmentation config/augmentation/efficientformer-detection.yaml\
#   --model config/model/efficientformer-detection.yaml\
#   --training config/training/efficientformer.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

  
#### HuggingFace datasets training
# To use HuggingFace datasets, you need to additionally install requirements-data.txt
# `pip install -r requirements-data.txt`
#### (END)

python train.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/resnet.yaml\
  --model config/model/resnet.yaml\
  --training config/training/resnet.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml


# python train.py\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/pidnet.yaml\
#   --model config/model/pidnet.yaml\
#   --training config/training/pidnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml
