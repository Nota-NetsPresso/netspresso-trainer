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

Please refer to [`scripts/example_train.sh`](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/scripts/example_train.sh).

NetsPresso Trainer is compatible with [NetsPresso](https://netspresso.ai/) service. We provide NetsPresso Trainer tutorial that contains whole procedure from model train to model compression and benchmark. Please refer to our [colab tutorial](https://colab.research.google.com/drive/1RBKMCPEa4x-4X31zqzTS8WgQI9TQt3e-?usp=sharing).