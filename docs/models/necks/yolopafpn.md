# YOLOPAFPN

YOLOPAFPN based on [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430).

YOLOPAFPN is a modified PAFPN for YOLOX model. Therefore, although YOLOPAFP is compatible with various backbones, we recommend to use it when constructing YOLOX models. The size is determined by `dep_mul` value, which defines the repetition of CSPLayers.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "yolopafpn" to use `YOLOPAFPN` neck. |
| `params.depthwise`| (bool) Whether to enable depthwise convolution for the `YOLOPAFPN` neck. |
| `params.dep_mul` | (int) Multiplying factor determining the repetition count of `CSPLayer` in the backbone. |
| `params.act_type` | (int) Type of activation function for the model. Supporting activation functions are described in [[here]](../../components/model/activations.md). |

## Model configuration examples
<details>
  <summary>PAFPN for YOLOX-nano</summary>
  
  ```yaml
  model:
    architecture:
      neck:
        name: yolopafpn
        params:
          depthwise: True
          dep_mul: 0.33
          act_type: "silu"
  ```
</details>


<details>
  <summary>PAFPN for YOLOX-s</summary>
  
  ```yaml
  model:
    architecture:
      neck:
        name: yolopafpn
        params:
          depthwise: False
          dep_mul: 0.33
          act_type: "silu"
  ```
</details>

## Related links
- [`Megvii-BaseDetection/YOLOX`](https://github.com/Megvii-BaseDetection/YOLOX)