# python -m torch.distributed.launch\
#   --nproc_per_node 2\
#   train_fx.py\
#   --data config/data/beans.yaml\
#   --augmentation config/augmentation/vit.yaml\
#   --model config/model/vit.yaml\
#   --training config/training/vit.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml\
#   --model-checkpoint classification_vit_fx.pt

python train_fx.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/mobilevit.yaml\
  --model config/model/mobilevit.yaml\
  --training config/training/mobilevit.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml\
  --model-checkpoint classification_mobilevit_fx.pt
