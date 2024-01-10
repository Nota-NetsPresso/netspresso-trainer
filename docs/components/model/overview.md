# Model - Overview

Netspresso Trainer provides a variety of backbones and heads, allowing flexible combinations. Users can choose appropriate backbones and heads based on their dataset and task requirements. The models can be optimized for on-device environments using Netspresso's compression and converting services.

We provide a configuration format which can easily construct backbones and heads to meet user requirements. As composed in the example of the ResNet50 model below, backbones and heads are structured as separate fields, and then connected. Also, Users can freely choose suitable loss modules suitable for the head and task. The range of supported models and the detailed configuration definitions for each model are extensively described in the separated [Models page](../../../models/overview).

```yaml
model:
  task: classification
  name: resnet50
  checkpoint: ./weights/resnet/resnet50.safetensors
  fx_model_checkpoint: ~
  resume_optimizer_checkpoint: ~
  freeze_backbone: False
  architecture:
    full:
    backbone:
    neck:
    head:
  losses:
    - criterion: cross_entropy
      label_smoothing: 0.1
      weight: ~
```

## Retraining the model from NetsPresso

If you have compressed model from NetsPresso, then it's time to retrain your model to get the best performance. Netspresso Trainer uses the same configuration format for retraining torch.fx GraphModule. This can be executed by specifying the path to the torch.fx model in the `fx_model_checkpoint` field. Since the torch.fx model file contains the complete model definition, fields like `architecture` become unnecessary, can be ignored.

```yaml
model:
  task: classification
  name: resnet50
  checkpoint: ~
  fx_model_checkpoint: ./path_to_your_fx_model.pt
  resume_optimizer_checkpoint: ~
  freeze_backbone: False
  architecture: # This field will be ignored since fx_model_checkpoint is activated
    ...
  losses:
    - criterion: cross_entropy
      label_smoothing: 0.1
      weight: ~
```



## Field list

| Field <img width=200/> | Description |
|---|---|
| `model.task` | (str) We support "classification", "segmentation", and "detection" now. |
| `model.name` | (str) A nickname to identify the model. |
| `model.checkpoint` | (str) Path to pretrained model weights. If there is no file on the path, automatically download from our storage. Only one of `checkpoint` or `fx_model_checkpoint` must have a value. |
| `model.fx_model_checkpoint` | (str) Path to `torch.fx` model file. This option can be used for models which compressed with NetsPresso service. Only one of `checkpoint` or `fx_model_checkpoint` must have a value. |
| `model.resume_optimizer_checkpoint` | (str) Path to resume optimizer and scheduler state from a checkpoint. |
| `model.freeze_backbone` | (bool) Whether to freeze backbone in training. |
| `model.architecture` | (dict) Detailed configuration of the model architecture. Please see [Model page](../../../models/overview) to find NetsPresso supporting models. |
| `model.losses` | (list) List of losses that model to learn. Please see [Losses page](../losses/) to find NetsPresso supporting loss modules. |
