#### HuggingFace datasets training
# To use HuggingFace datasets, you need to additionally install requirements-data.txt
# `pip install -r requirements-data.txt`
#### (END)


#### Classification

netspresso-train\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --training config/training/classification.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml

# netspresso-train\
#   --data config/data/beans.yaml\
#   --augmentation config/augmentation/classification.yaml\
#   --model config/model/efficientformer/efficientformer-l1-classification.yaml\
#   --training config/training/classification.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# netspresso-train\
#   --data config/data/beans.yaml\
#   --augmentation config/augmentation/classification.yaml\
#   --model config/model/mobilevit/mobilevit-s-classification.yaml\
#   --training config/training/classification.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# netspresso-train\
#   --data config/data/beans.yaml\
#   --augmentation config/augmentation/classification.yaml\
#   --model config/model/segformer/segformer-classification.yaml\
#   --training config/training/classification.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# netspresso-train\
#   --data config/data/beans.yaml\
#   --augmentation config/augmentation/classification.yaml\
#   --model config/model/vit/vit-classification.yaml\
#   --training config/training/classification.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

#### (END)


#### Segmentation

# netspresso-train\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/segmentation.yaml\
#   --model config/model/efficientformer/efficientformer-l1-segmentation.yaml\
#   --training config/training/segmentation.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# netspresso-train\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/segmentation.yaml\
#   --model config/model/resnet50/resnet50-segmentation.yaml\
#   --training config/training/segmentation.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# netspresso-train\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/segmentation.yaml\
#   --model config/model/segformer/segformer-segmentation.yaml\
#   --training config/training/segmentation.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# netspresso-train\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/segmentation.yaml\
#   --model config/model/pidnet/pidnet-s-segmentation.yaml\
#   --training config/training/segmentation.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

#### (END)


#### Detection

# netspresso-train\
#   --data config/data/traffic-sign.yaml\
#   --augmentation config/augmentation/detection.yaml\
#   --model config/model/efficientformer/efficientformer-l1-detection.yaml\
#   --training config/training/detection.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

#### (END)


#### Multi-GPU training
# Put the number of GPU(s) to use in training at `--nproc_per_node`
#### (END)

# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/chess.yaml\
#   --augmentation config/augmentation/classification.yaml\
#   --model config/model/resnet/resnet50-classification.yaml\
#   --training config/training/classification.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml


# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/segmentation.yaml\
#   --model config/model/pidnet/pidnet-s-segmentation.yaml\
#   --training config/training/segmentation.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml


# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/traffic-sign.yaml\
#   --augmentation config/augmentation/detection.yaml\
#   --model config/model/efficientformer/efficientformer-l1-detection.yaml\
#   --training config/training/detection.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml