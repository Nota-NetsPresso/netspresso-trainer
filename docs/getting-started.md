## Getting started

Write your training script in `train.py` like:

```python
from netspresso_trainer import set_arguments, train

args_parsed, args = set_arguments(is_graphmodule_training=False)
train(args_parsed, args, is_graphmodule_training=False)
```

Then, train your model with your own configuraiton:

```bash
python train.py\
  --data config/data/beans.yaml\
  --augmentation config/augmentation/resnet.yaml\
  --model config/model/resnet.yaml\
  --training config/training/resnet.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

Please refer to [`example_train.sh`](./example_train.sh) and [`example_train_fx.sh`](./example_train_fx.sh).