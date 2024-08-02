# CSPDarkNet

CSPDarkNet backbone based on [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430).

CSPDarkNet is a modified model from Darknet53 by adopting the strategy of CSPNet. Therefore, the structure of the model is fixed, and neither the number of stages nor type of blocks can be changed. The size of the model is determined by two values, which define the feature dimensions within the model and the repetition of CSPLayers.

## Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "cspdarknet" to use `CSPDarkNet` backbone. |
| `params.depthwise`| (bool) Whether to enable depthwise convolution for the `CSPDarkNet` backbone. |
| `params.dep_mul` | (float) Multiplying factor determining the repetition count of `CSPLayer` in the backbone. |
| `params.wid_mul` | (float) Multiplying factor adjusting the input/output dimensions of convolutional layers throughout the backbone. |
| `params.act_type` | (str) Type of activation function for the model. Supporting activation functions are described in [[here]](../../components/model/activations.md). |

## Model configuration examples
<details>
  <summary>CSPDarkNet-nano</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: cspdarknet
        params:
          depthwise: True
          dep_mul: &dep_mul 0.33
          wid_mul: 0.25
          act_type: &act_type "silu"
        stage_params: ~
  ```
</details>

<details>
  <summary>CSPDarkNet-tiny</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: cspdarknet
        params:
          depthwise: False
          dep_mul: &dep_mul 0.33
          wid_mul: 0.375
          act_type: &act_type "silu"
        stage_params: ~
  ```
</details>


<details>
  <summary>CSPDarkNet-s</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: cspdarknet
        params:
          depthwise: False
          dep_mul: &dep_mul 0.33
          wid_mul: 0.5
          act_type: &act_type "silu"
        stage_params: ~
  ```
</details>

<details>
  <summary>CSPDarkNet-m</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: cspdarknet
        params:
          depthwise: False
          dep_mul: &dep_mul 0.67
          wid_mul: 0.75
          act_type: &act_type "silu"
        stage_params: ~
  ```
</details>

<details>
  <summary>CSPDarkNet-l</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: cspdarknet
        params:
          depthwise: False
          dep_mul: &dep_mul 1.0
          wid_mul: 1.0
          act_type: &act_type "silu"
        stage_params: ~
  ```
</details>

<details>
  <summary>CSPDarkNet-x</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: cspdarknet
        params:
          depthwise: False
          dep_mul: &dep_mul 1.33
          wid_mul: 1.25
          act_type: &act_type "silu"
        stage_params: ~
  ```
</details>

## Related links
- [`Megvii-BaseDetection/YOLOX`](https://github.com/Megvii-BaseDetection/YOLOX)