python train_fx.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --training config/training/classification.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml\
  --fx-model-checkpoint classification_resnet_fx.pt