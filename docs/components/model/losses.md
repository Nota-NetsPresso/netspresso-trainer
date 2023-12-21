# Losses

Loss modules is very important in the training of neural networks, because it actually guides and shapes the learning process by minimizing the loss. It can affect the speed of convergence during training and the overall robustness of the model. 

However, loss functions can vary depending on the tasks, especially in tasks like detection and segmentation, where specialized loss functions are required. Therefore, NetsPresso Trainer provides a predefined variety of loss modules, designed for flexible use across different tasks. Users can seamlessly apply the appropriate loss function to their desired task through simple configuration settings.


## Supporting loss modules

### CrossEntropy

Cross entropy loss. This loss follows the nn.CrossEntropyLoss in torch library.

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "cross_entropy" to use CrossEntropy loss. |
| `params.weight` | (float) Weight for this cross entropy loss. |

### SigmoidFocalLoss

Focal loss based on [Focal loss for dense object detections](). This loss follows the [sigmoid_focal_loss](https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html) in the torch library.

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "focal_loss" to use SigmoidFocalLoss loss. |
| `params.weight` | (float) Weight for this focal loss. |
| `params.alpha` | (float) Balancing parameter alpha for focal loss. |
| `params.gamma` | (float) Focusing parameter gamma for focal loss. |

### YOLOXLoss

Loss module for [AnchorFreeDecoupledHead](). This loss contains ...

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "yolox_loss" to use YOLOXLoss loss. |
| `params.weight` | (float) Weight for this YOLOX loss. |

### RetinaNetLoss 

Loss module for [AnchorDecoupledHead](). This loss contains classification loss by focal loss and box regression loss by L1 loss.

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "retinanet_loss" to use RetinaNetLoss loss. |
| `params.weight` | (float) Weight for this RetinaNet loss. |

### PIDNetLoss

Loss for [PIDNet](). This loss contains ...

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "pidnet_loss" to use PIDNetLoss loss. |
| `params.weight` | (float) Weight for this PIDNet loss. |
| `params.ignore_index` | (int) |
