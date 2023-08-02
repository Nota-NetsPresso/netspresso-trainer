python train_fx.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/mobilevit.yaml\
  --model config/model/mobilevit.yaml\
  --training config/training/mobilevit.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml\
  --model-checkpoint classification_mobilevit_fx.pt


# python train_fx.py\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/pidnet.yaml\
#   --model config/model/pidnet.yaml\
#   --training config/training/pidnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml\
#   --model-checkpoint segmentation_pidnet_fx.pt


#### Multi-GPU training
# Put the number of GPU(s) to use in training at `--nproc_per_node`
#### (END)

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


# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/sidewalk-semantic.yaml\
#   --augmentation config/augmentation/pidnet.yaml\
#   --model config/model/pidnet.yaml\
#   --training config/training/pidnet.yaml\
#   --logging config/logging.yaml\
#   --environment config/environment.yaml
