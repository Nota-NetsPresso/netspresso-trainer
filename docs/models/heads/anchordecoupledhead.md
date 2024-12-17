# AnchorDecoupledHead

Decoupled detection head with anchors based on [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

We have named the detection head of RetinaNet as AnchorDecoupledHead to represent it in a more general term. AnchorDecoupledHead consists of a box regression head and a classification head for the given intermediate features, predicting detection boxes for the anchors at each feature's pixel location. Additionally, we provide the option to adjust the number of convolution layers used in the heads through the `num_layers` value, it is equivalent with RetinaNet when set to 4.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "retinanet_head" to use `RetinaNetHead` head. |
| `params.anchor_sizes` | (list[list[int]]) Default anchor sizes for each intermediate feature. |
| `params.aspect_ratios` | (list[float]) List of aspect ratio for each anchor. |
| `params.num_layers` | (int) The number of convolution layers of regression and classification head. |
| `params.norm_layer` | (str) Normalization type for the head. |

## Model configuration example

<details>
  <summary>Anchor-based decoupled detection head</summary>
  
  ```yaml
  model:
    architecture:
      head:
        name: anchor_decoupled_head
        params:
          anchor_sizes: [[32,], [64,], [128,], [256,]]
          aspect_ratios: [0.5, 1.0, 2.0]
          num_layers: 1
          norm_type: batch_norm 
  ```
</details>

## Related links
- [`pytorch/vision`](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py)