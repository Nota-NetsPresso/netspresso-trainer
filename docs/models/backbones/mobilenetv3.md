# MobileNetV3

MobileNetV3 backbone based on [Searching for MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf).

We provide a configuration that allows users to define each inverted residual block in MobileNetV3 individually. You can specify in a list format the number and form of each inverted residual block for every stage. Using this, we provide both MobileNetV3-small and MobileNetV3-large.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "mobilenetv3" to use `MobileNetV3` backbone. |
| `stage_params[n].in_channels` | (list[int]) Input dimensions for the inverted residual blocks in the stage. |
| `stage_params[n].kernel_sizes` | (list[int]) Convolution kernel sizes for the inverted residual blocks in the stage. |
| `stage_params[n].expanded_channels` | (list[int]) Expanded dimensions for the inverted residual blocks in the stage. |
| `stage_params[n].out_channels` | (list[int]) Output dimensions for the inverted residual blocks in the stage. |
| `stage_params[n].use_se` | (list[bool]) Flags that determine whether to use squeeze-and-excitation blocks for the inverted residual blocks in the stage. |
| `stage_params[n].activation` | (list[str]) Type of activation functions for the inverted residual blocks in the stage. Supporting activation functions are described in [[here]](../../components/model/activations.md) |
| `stage_params[n].stride` | (list[int]) Stride values for the inverted residual blocks included in the stage. |
| `stage_params[n].dilation` | (list[int]) Dilation values for the inverted residual blocks in the stage. |

## Model configuration examples

<details>
    <summary>MobileNetV3-small</summary>

    ```yaml
    model:
      architecture:
        backbone:
          name: mobilenetv3
          params: ~
          stage_params:
            -
              in_channels: [16]
              kernel_sizes: [3]
              expanded_channels: [16]
              out_channels: [16]
              use_se: [True]
              act_type: ["relu"]
              stride: [2]
            -
              in_channels: [16, 24]
              kernel_sizes: [3, 3]
              expanded_channels: [72, 88]
              out_channels: [24, 24]
              use_se: [False, False]
              act_type: ["relu", "relu"]
              stride: [2, 1]
            -
              in_channels: [24, 40, 40, 40, 48]
              kernel_sizes: [5, 5, 5, 5, 5]
              expanded_channels: [96, 240, 240, 120, 144]
              out_channels: [40, 40, 40, 48, 48]
              use_se: [True, True, True, True, True]
              act_type: ["hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"]
              stride: [2, 1, 1, 1, 1]
            -
              in_channels: [48, 96, 96]
              kernel_sizes: [5, 5, 5]
              expanded_channels: [288, 576, 576]
              out_channels: [96, 96, 96]
              use_se: [True, True, True]
              act_type: ["hard_swish", "hard_swish", "hard_swish"]
              stride: [2, 1, 1]
    ```
</details>

<details>
  <summary>MobileNetV3-large</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilenetv3
        params: ~
        stage_params:
          -
            in_channels: [16, 16, 24]
            kernel_sizes: [3, 3, 3]
            expanded_channels: [16, 64, 72]
            out_channels: [16, 24, 24]
            use_se: [False, False, False]
            act_type: ["relu", "relu", "relu"]
            stride: [1, 2, 1]
          - 
            in_channels: [24, 40, 40]
            kernel_sizes: [5, 5, 5]
            expanded_channels: [72, 120, 120]
            out_channels: [40, 40, 40]
            use_se: [True, True, True]
            act_type: ["relu", "relu", "relu"]
            stride: [2, 1, 1]
          -
            in_channels: [40, 80, 80, 80, 80, 112]
            kernel_sizes: [3, 3, 3, 3, 3, 3]
            expanded_channels: [240, 200, 184, 184, 480, 672]
            out_channels: [80, 80, 80, 80, 112, 112]
            use_se: [False, False, False, False, True, True]
            act_type: ["hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"]
            stride: [2, 1, 1, 1, 1, 1]
          -
            in_channels: [112, 160, 160]
            kernel_sizes: [5, 5, 5]
            expanded_channels: [672, 960, 960]
            out_channels: [160, 160, 160]
            use_se: [True, True, True]
            act_type: ["hard_swish", "hard_swish", "hard_swish"]
            stride: [2, 1, 1]
  ```
</details>

## Related links
- [`pytorch/vision`](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)