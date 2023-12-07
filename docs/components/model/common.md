# Common

Netspresso Trainer provides a variety of backbones and heads, allowing for flexible combinations. Users can choose appropriate backbones and heads based on their dataset and task requirements. The models can be optimized for on-device environments using Netspresso's compression and converting services.

## Field list

### Common

| Field <img width=200/> | Description |
|---|---|
| `task` | (str) `classification` for image classification, `segmentation` for semantic segmentation, and `detection` for object detectionn |
| `name` | (str) A nickname to identify the model. |
| `checkpoint` | (str) Path to pretrained model weights. If there is no file on the path, automatically download from our storage. Only one of `checkpoint` or `fx_model_checkpoint` must have a value. |
| `fx_model_checkpoint` | (str) Path to `torch.fx` model file. This option can be used for models which compressed with NetsPresso service. Only one of `checkpoint` or `fx_model_checkpoint` must have a value. |
| `resume_optimizer_checkpoint` | (str) Path to resume optimizer and scheduler state from a checkpoint. |
| `freeze_backbone` | (bool) Whether to freeze backbone in training. |
| `architecture` | (dict) Detailed configuration of the model's architecture. |
| `losses` | (list) List of losses that model to learn. Detailed configuration of losses are described in [loss components page](../losses/). |

### Architecture

| Field <img width=200/> | Description |
|---|---|
| `architecture.full` | (dict) Configuration for the full model. |
| `architecture.backbone` | (dict) Configuration for the backbone component. |
| `architecture.neck` | (dict) Configuration for the neck component. |
| `architecture.head` | (dict) Configuration for the head component. |

### Full model configuration

| Field <img width=200/> | Description |
|---|---|
| `architecture.full` | (dict) |

### Model backbone configuration

| Field <img width=200/> | Description |
|---|---|
| `architecture.backbone.name` | (str) Name of the backbone component. |
| `architecture.backbone.params` | (dict) Parameters about the chosen backbone. Detailed configuration of backbones are described in [backbone components page](../backbones) |
| `architecture.backbone.stage_params` | (list) Parameters that changes across the intermediate stages about the chosen backbone. Detailed configuration of backbones are described in [backbone components page](../backbones) |

### Model head configuration

| Field <img width=200/> | Description |
|---|---|
| `architecture.head.name` | (str) Name of the head component. |
| `architecture.head.params` | (dict) Parameters about the chosen head. Detailed configuration of heads are described in [head components page](../heads/) |