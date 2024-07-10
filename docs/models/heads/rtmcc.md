# RTMCC

RTMCC head based on [RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose](https://arxiv.org/abs/2303.07399).

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "rtmcc" to use `RTMCC` head. |
| `params.conv_kernel` | (int) Kernel size of convolution layer. |
| `params.attention_channels` | (int) Dimension of gated attention unit. |
| `params.attention_act_type` | (str) Activation type of gated attention unit. |
| `params.attention_pos_enc` | (bool) Whether to use rotary position embedding for gated attention unit. |
| `params.s` | (int) Self attention feature dimension of gated attention unit. |
| `params.expansion_factor` | (int) Expansion factor of gated attention unit.  |
| `params.dropout_rate` | (float) Dropout rate of gated attention unit. |
| `params.drop_path` | (float) Drop path rate of gated attention unit. |
| `params.use_rel_bias` | (bool) Whether to use relative bias for gated attention unit. |
| `params.simcc_split_ratio` | (float) Split ratio of pixels. |
| `params.target_size` | (list) Original input image size. |
| `params.backbone_stride` | (int) Stride of input feature from original image produced by backbone. |

## Model configuration example

<details>
  <summary>RTMCC head</summary>
  
  ```yaml
  model:
    architecture:
      head:
      name: rtmcc
      params:
        conv_kernel: 7
        attention_channels: 256
        attention_act_type: 'silu'
        attention_pos_enc: False
        s: 128
        expansion_factor: 2
        dropout_rate: 0.
        drop_path: 0.
        use_rel_bias: False
        simcc_split_ratio: 2.
        target_size: [256, 256]
        backbone_stride: 32
  ```
</details>

## Related links
- [`RTMPose`](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)