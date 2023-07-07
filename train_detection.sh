# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/coco_yolo.yaml\
#   --augmentation config/augmentation/efficientformer-detection.yaml\
#   --model config/model/efficientformer-detection.yaml\
#   --training config/training/efficientformer.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

python train.py\
  --data config/data/traffic-sign.yaml\
  --augmentation config/augmentation/efficientformer-detection.yaml\
  --model config/model/efficientformer-detection.yaml\
  --training config/training/efficientformer.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml