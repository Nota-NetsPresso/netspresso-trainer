# python train.py\
#   --data config/data/chess.yaml\
#   --augmentation config/augmentation/resnet.yaml\
#   --model config/model/resnet.yaml\
#   --training config/training/resnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml

# python -m torch.distributed.launch
#   --nproc_per_node 2\
#   train.py\
#   --data config/data/chess.yaml\
#   --augmentation config/augmentation/resnet.yaml\
#   --model config/model/resnet.yaml\
#   --training config/training/resnet.yaml\
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
