# Model - Overview

Netspresso Trainer provides a variety of backbones and heads, allowing flexible combinations. Users can choose appropriate backbones and heads based on their dataset and task requirements. The models can be optimized for on-device environments using Netspresso's compression and converting services.

We provide a configuration format which can easily construct backbones and heads to meet user requirements. As composed in the example of the ResNet50 model below, backbones and heads are structured as separate fields, and then connected. Also, Users can freely choose suitable loss modules suitable for the head and task. The range of supported models and the detailed configuration definitions for each model are extensively described in the separated [Models page](../../../models/overview).

```yaml
model:
  task: classification
  name: resnet50
  checkpoint:
    use_pretrained: True
    load_head: False
    path: ~
    optimizer_path: ~
  freeze_backbone: False
  architecture:
    full: ...
    backbone: ...
    neck: ...
    head: ...
  postprocessor: ~
  losses:
    - criterion: cross_entropy
      label_smoothing: 0.1
      weight: ~
```

## Retraining the model from NetsPresso

If you have compressed model from NetsPresso, then it's time to retrain your model to get the best performance. Netspresso Trainer uses the same configuration format for retraining torch.fx GraphModule. This can be executed by specifying the path to the torch.fx model in the `path`. The torch.fx model must have .pt extension which indicating it is a torch.fx model (In NetsPresso Trainer, vanilla torch model weights file has .safetensors extension). Since the torch.fx model file contains the complete model definition, fields like `architecture` become unnecessary, can be ignored.

```yaml
model:
  task: classification
  name: resnet50
  checkpoint:
    use_pretrained: ~ # This field will be ignored
    load_head: ~ # This field will be ignored
    path: ./path_to_your_fx_model.pt # The torch.fx model must have .pt extension which indicating it is a torch.fx model
    optimizer_path: ~ 
  freeze_backbone: False
  architecture: ~ # This field will be ignored
  postprocessor: ~
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
| `model.checkpoint.use_pretrained` | (bool) Whether to use the pretrained checkpoint. At first time, you will download and save the pretrained checkpoint. |
| `model.checkpoint.load_head` | (bool) Whether to use the pretrained checkpoint for `head` module. |
| `model.checkpoint.path` | (str) Checkpoint path to resume training. If `None` and `use_pretrained` is `False`, you can train you model from scratch. |
| `model.checkpoint.optimizer_path` | (str) Optimizer checkpoint path for resuming training. |
| `model.freeze_backbone` | (bool) Whether to freeze backbone in training. |
| `model.architecture` | (dict) Detailed configuration of the model architecture. Please see [Model page](../../../models/overview) to find NetsPresso supporting models. |
| `model.postprocessor` | (dict) Detailed configuration of the model postprocessor. Please see [Postprocessor page](../postprocessor/)|
| `model.losses` | (list) List of losses that model to learn. Please see [Losses page](../losses/) to find NetsPresso supporting loss modules. |
