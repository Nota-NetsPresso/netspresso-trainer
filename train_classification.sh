python train.py\
  --data config/data/chess.yaml\
  --config config/models/resnet.yaml\
  --training config/training/resnet.yaml

# python -m torch.distributed.launch
#   --nproc_per_node 2\
#   train.py\
#   --data config/data/chess.yaml\
#   --config config/models/resnet.yaml\
#   --training config/training/resnet.yaml

#### HuggingFace datasets training
# To use HuggingFace datasets, you need to additionally install requirements-data.txt
# `pip install -r requirements-data.txt`
#### (END)

# python train.py\
#   --data config/data/beans.yaml\
#   --config config/models/resnet.yaml\
#   --training config/training/resnet.yaml
