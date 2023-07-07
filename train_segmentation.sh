# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/pidnet.yaml\
#   --model config/model/pidnet.yaml\
#   --training config/training/pidnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

python train.py\
  --data config/data/sidewalk-semantic.yaml\
  --augmentation config/augmentation/pidnet.yaml\
  --model config/model/pidnet.yaml\
  --training config/training/pidnet.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml