## Training

Write your training script in `train.py` like:

```python
from netspresso_trainer import train_cli

if __name__ == '__main__':
    logging_dir = train_cli()
    print(f"Training results are saved at: {logging_dir}")
```

Then, train your model with your own configuraiton:

```bash
python train.py\
  --data config/data/huggingface/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --training config/training.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

Or you can start NetsPresso Trainer by just executing console script which has same feature.

```bash
netspresso-train\
  --data config/data/huggingface/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --training config/training.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

Please refer to [`scripts/example_train.sh`](https://github.com/Nota-NetsPresso/netspresso-trainer/blob/dev/scripts/example_train.sh).

NetsPresso Trainer is compatible with [NetsPresso](https://netspresso.ai/) service. We provide NetsPresso Trainer tutorial that contains whole procedure from model train to model compression and benchmark. Please refer to our [colab tutorial](https://colab.research.google.com/drive/1RBKMCPEa4x-4X31zqzTS8WgQI9TQt3e-?usp=sharing).

## Evaluation

Write your evaluation script in `evaluation.py` like:

```bash
from netspresso_trainer import evaluation_cli

if __name__ == '__main__':
    logging_dir = evaluation_cli()
    
    print(f"Evaluation results are saved at: {logging_dir}")
```

Then, evaluate your model with your own configuraiton:

```bash
python evaluation.py\
  --data config/data/huggingface/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```

## Inference

Write your inference script in `inference.py` like:

```bash
from netspresso_trainer import inference_cli

if __name__ == '__main__':
    logging_dir = inference_cli()
    
    print(f"Inference results are saved at: {logging_dir}")
```

Then, inference your dataset:

```bash
python inference.py\
  --data config/data/huggingface/beans.yaml\
  --augmentation config/augmentation/classification.yaml\
  --model config/model/resnet/resnet50-classification.yaml\
  --logging config/logging.yaml\
  --environment config/environment.yaml
```