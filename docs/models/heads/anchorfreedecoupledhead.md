# AnchorFreeDecoupledHead

Anchor-free decoupled detection head based on [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430).

We provide the head of YOLOX as AnchorFreeDecoupledHead. There are no differnece with the original model, and currently, it is set to pass non-maximum suppression function.

## Compatibility matrix

<table>
  <tr>
    <th>Supporting backbones</th>
    <th>Supporting necks</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
  </tr>
  <tr>
    <td>
      ResNet<br />
      MobileNetV3<br />
      MixNet<br />
      CSPDarkNet<br />
      MobileViT<br />
      MixTransformer<br />
      EfficientFormer
    </td>
    <td>
    FPN<br />
    YOLOPAFPN<br />
    </td>
    <td>Supported</td>
    <td>Supported</td>
  </tr>
</table>

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "yolox_head" to use `YOLOX` head. |
| `params.act_type` | (float) Activation function for the head. |
| `params.score_thresh` | (float) Score thresholding value applied during the decoding step. |
| `params.class_agnostic` | (bool) Whether to process class-agnostic non-maximum suppression. |

## Model configuration example

<details>
  <summary>Anchor-free decoupled detection head</summary>
  
  ```yaml
  model:
    architecture:
      head:
        name: anchor_free_decoupled_head
        params:
          act_type: "silu"
    postprocessor:
      score_thresh: 0.7
      nms_thresh: 0.45
      class_agnostic: False
  ```
</details>

## Related links
- [`Megvii-BaseDetection/YOLOX`](https://github.com/Megvii-BaseDetection/YOLOX)