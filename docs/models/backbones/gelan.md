# GELAN (Generalized Efficient Layer Aggregation Network)

GELAN backbone based on [YOLOv9: Learning What You Want to Learn
Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616).

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "gelan" to use `GELAN` backbone. |
| `stage_params[n] case1: Conv2D` | (list) Build `Conv2D` layer under following format: `['conv', out_channels, kernel_size, stride]`. |
| `stage_params[n] case2: ELAN` | (list) Build `ELAN` block under following format: `['elan', out_channels, part_channels, use_identity]`. |
| `stage_params[n] case3: RepNCSPELAN` | (list) Build `RepNCSPELAN` block under following format: `['repncspelan', out_channels, part_channels, use_identity, depth]`. |
| `stage_params[n] case4: AConv` | (list) Build `AConv` block under following format: `['aconv', out_channels]`. |
| `stage_params[n] case4: ADown` | (list) Build `ADown` block under following format: `['adown', out_channels]`. |

## Model configuration examples
<details>
    <summary>GELAN-tiny</summary>

  ```yaml
  model:
    architecture:
      backbone:
        name: gelan
        params:
          stem_out_channels: 16
          stem_kernel_size: 3
          stem_stride: 2
          return_stage_idx: ~
          act_type: &act_type silu
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # ELAN: ['elan', out_channels, part_channels, use_identity]
          # RepNCSPELAN: ['repncspelan', out_channels, part_channels, use_identity, depth]
          # AConv: ['aconv', out_channels]
          # ADown: ['adown', out_channels]
          -
            - ['conv', 32, 3, 2]
            - ['elan', 32, 32, false]
          -
            - ['aconv', 64]
            - ['repncspelan', 64, 64, false, 3]
          -
            - ['aconv', 96]
            - ['repncspelan', 96, 96, false, 3]
          - 
            - ['aconv', 128]
            - ['repncspelan', 128, 128, false, 3]
  ```
</details>

<details>
    <summary>GELAN-s</summary>

  ```yaml
  model:
    architecture:
      backbone:
        name: gelan
        params:
          stem_out_channels: 32
          stem_kernel_size: 3
          stem_stride: 2
          return_stage_idx: ~
          act_type: &act_type silu
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # ELAN: ['elan', out_channels, part_channels, use_identity]
          # RepNCSPELAN: ['repncspelan', out_channels, part_channels, use_identity, depth]
          # AConv: ['aconv', out_channels]
          # ADown: ['adown', out_channels]
          -
            - ['conv', 64, 3, 2]
            - ['elan', 64, 64, false]
          -
            - ['aconv', 128]
            - ['repncspelan', 128, 128, false, 3]
          -
            - ['aconv', 192]
            - ['repncspelan', 192, 192, false, 3]
          - 
            - ['aconv', 256]
            - ['repncspelan', 256, 256, false, 3]
  ```
</details>

<details>
    <summary>GELAN-m</summary>

  ```yaml
  model:
    architecture:
      backbone:
        name: gelan
        params:
          stem_out_channels: 32
          stem_kernel_size: 3
          stem_stride: 2
          return_stage_idx: ~
          act_type: &act_type silu
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # ELAN: ['elan', out_channels, part_channels, use_identity]
          # RepNCSPELAN: ['repncspelan', out_channels, part_channels, use_identity, depth]
          # AConv: ['aconv', out_channels]
          # ADown: ['adown', out_channels]
          -
            - ['conv', 64, 3, 2]
            - ['repncspelan', 128, 128, false, 1]
          -
            - ['aconv', 240]
            - ['repncspelan', 240, 240, false, 1]
          -
            - ['aconv', 360]
            - ['repncspelan', 360, 360, false, 1]
          - 
            - ['aconv', 480]
            - ['repncspelan', 480, 480, false, 1]
  ```
</details>

<details>
    <summary>GELAN-c</summary>

  ```yaml
  model:
    architecture:
      backbone:
        name: gelan
        params:
          stem_out_channels: 64
          stem_kernel_size: 3
          stem_stride: 2
          return_stage_idx: ~
          act_type: &act_type silu
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # ELAN: ['elan', out_channels, part_channels, use_identity]
          # RepNCSPELAN: ['repncspelan', out_channels, part_channels, use_identity, depth]
          # AConv: ['aconv', out_channels]
          # ADown: ['adown', out_channels]
          -
            - ['conv', 128, 3, 2]
            - ['repncspelan', 256, 128, false, 1]
          -
            - ['adown', 256]
            - ['repncspelan', 512, 256, false, 1]
          -
            - ['adown', 512]
            - ['repncspelan', 512, 512, false, 1]
          - 
            - ['adown', 512]
            - ['repncspelan', 512, 512, false, 1]
  ```
</details>