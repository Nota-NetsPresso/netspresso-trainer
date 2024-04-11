# Losses

Loss modules are very important in the training of neural networks, because as they guide and shape the learning process by minimizing the loss. They can affect the speed of convergence during training and the overall robustness of the model. 

However, loss functions can vary depending on the task, especially in tasks like detection and segmentation, where specialized loss functions are required. Therefore, NetsPresso Trainer provides a predefined variety of loss modules, designed for flexible use across different tasks. Users can seamlessly apply the appropriate loss function to their desired task through simple configuration settings.


## Supporting loss modules

The currently supported methods in NetsPresso Trainer are as follows. Since techniques are adapted from pre-existing codes, hence most of the parameters remain unchanged. We note that most of these parameter descriptions are derived from original implementations.

We appreciate all the original code owners and we also do our best to make other values.

### CrossEntropyLoss

Cross entropy loss. This loss follows the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "cross_entropy" to use `CrossEntropyLoss`. |
| `label_smoothing` | (float) Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. |
| `weight` | (float) Weight for this cross entropy loss. |

<details>
  <summary>Cross entropy loss example</summary>
  ```yaml
  model:
    losses:
      - criterion: cross_entropy
        label_smoothing: 0.1
        weight: ~
  ```
</details>

### SigmoidFocalLoss

Focal loss based on [Focal loss for dense object detections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf). This loss follows the [sigmoid_focal_loss](https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html) in the torch library.

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "focal_loss" to use `SigmoidFocalLoss`. |
| `alpha` | (float) Balancing parameter alpha for focal loss. |
| `gamma` | (float) Focusing parameter gamma for focal loss. |
| `weight` | (float) Weight for this focal loss. |

<details>
  <summary>Focal loss example</summary>
  ```yaml
  model:
    losses:
      - criterion: focal_loss
        alpha: 0.25
        gamma: 2.0
        weight: ~
  ```
</details>

### YOLOXLoss

Loss module for [AnchorFreeDecoupledHead](../../models/heads/anchorfreedecoupledhead.md). This loss follows the [YOLOX implementation](https://github.com/Megvii-BaseDetection/YOLOX).

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "yolox_loss" to use `YOLOXLoss`. |
| `weight` | (float) Weight for this YOLOX loss. |
| `l1_activate_epoch` | (int) Activate l1 loss at `l1_activate_epoch` epoch. |

<details>
  <summary>YOLOX loss example</summary>
  ```yaml
  model:
    losses:
      - criterion: yolox_loss
        weight: ~
        l1_activate_epoch: 1
  ```
</details>

### RetinaNetLoss 

Loss module for [AnchorDecoupledHead](../../models/heads/anchordecoupledhead.md). This loss follows torchvision implementation, it contains classification loss via focal loss and box regression loss via L1 loss.

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "retinanet_loss" to use `RetinaNetLoss`. |
| `weight` | (float) Weight for this RetinaNet loss. |

<details>
  <summary>RetinaNetLoss loss example</summary>
  ```yaml
  model:
    losses:
      - criterion: retinanet_loss
        weight: ~
  ```
</details>

### PIDNetLoss

Loss module for PIDNet. This loss follows [official implementation repository](https://github.com/XuJiacong/PIDNet).

| Field <img width=200/> | Description |
|---|---|
| `criterion` | (str) Criterion must be "pidnet_loss" to use `PIDNetLoss`. |
| `weight` | (float) Weight for this PIDNet loss. |
| `ignore_index` | (int) A target value that is ignored and does not contribute to the input gradient. |

<details>
  <summary>PIDNetLoss loss example</summary>
  ```yaml
  model:
    losses:
      - criterion: pidnet_loss
        weight: ~
        ignore_index: 255
  ```
</details>
