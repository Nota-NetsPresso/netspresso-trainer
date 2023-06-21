# python -m torch.distributed.launch
#   --nproc_per_node 2\
#   train.py\
#   --data config/data/chess.yaml\
#   --config config/models/resnet.yaml\
#   --training config/training/resnet.yaml

python train.py\
  --data config/data/chess.yaml\
  --config config/models/resnet.yaml\
  --training config/training/resnet.yaml