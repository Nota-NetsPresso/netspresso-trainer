# MobileNetV4

MobileNetV4 backbone based on [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518).

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "mobilenetv4" to use `MobileNetV4` backbone. |
| `stage_params[n] case1: Conv2D` | (list) Build `Conv2D` layer under following format: `['conv', out_channels, kernel_size, stride]`. |
| `stage_params[n] case2: FusedIB` | (list) Build `FusedIB` block under following format: `['fi', out_channels, hidden_channels, kernel_size, stride]`. |
| `stage_params[n] case3: UniversalInvertedResidualBlock` | (list) Build `UniversalInvertedResidualBlock` block under following format: `['uir', out_channels, hidden_channels, extra_dw, extra_dw_kernel_size, middle_dw, middle_dw_kernel_size, stride]`. |
| `stage_params[n] case4: MobileMultiQueryAttention2D` | (list) Build `MobileMultiQueryAttention2D` block under following format: `['mmqa', out_channels, attention_channel, num_attention_heads, query_pooling_stride, key_val_downsample, key_val_downsample_kernel_size, key_val_downsample_stride, stride]`. |

## Model configuration examples

<details>
    <summary>MobileNetV4-conv-small</summary>

    ```yaml
    model:
      architecture:
        backbone:
          name: mobilenetv4
          params:
            stem_out_channel: 32
            stem_kernel_size: 3
            stem_stride: 2
            final_conv_out_channel: 960
            final_conv_kernel_size: 1
            final_conv_stride: 1
            norm_type: batch_norm
            act_type: relu
            return_stage_idx: ~
            layer_scale: 0.1
          stage_params:
            # Conv2D: ['conv', out_channels, kernel_size, stride]
            # FusedIB: ['fi', out_channels, hidden_channels, kernel_size, stride]
            # UniversalInvertedResidualBlock: ['uir', out_channels, hidden_channels, extra_dw, extra_dw_kernel_size, middle_dw, middle_dw_kernel_size, stride]
            # MobileMultiQueryAttention2D: ['mmqa', out_channels, attention_channel, num_attention_heads, query_pooling_stride, key_val_downsample, key_val_downsample_kernel_size, key_val_downsample_stride, stride]
            - 
              - ['conv', 32, 3, 2]
              - ['conv', 32, 1, 1]
            - 
              - ['conv', 96, 3, 2]
              - ['conv', 64, 1, 1]
            - 
              - ['uir', 96, 192, True, 5, True, 5, 2]
              - ['uir', 96, 192, False, ~, True, 3, 1]
              - ['uir', 96, 192, False, ~, True, 3, 1]
              - ['uir', 96, 192, False, ~, True, 3, 1]
              - ['uir', 96, 192, False, ~, True, 3, 1]
              - ['uir', 96, 384, True, 3, False, ~, 1]
            - 
              - ['uir', 128, 576, True, 3, True, 3, 2]
              - ['uir', 128, 512, True, 5, True, 5, 1]
              - ['uir', 128, 512, False, ~, True, 5, 1]
              - ['uir', 128, 384, False, ~, True, 5, 1]
              - ['uir', 128, 512, False, ~, True, 3, 1]
              - ['uir', 128, 512, False, ~, True, 3, 1]
    ```
</details>

<details>
  <summary>MobileNetV4-conv-medium</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilenetv4
        params:
          stem_out_channel: 32
          stem_kernel_size: 3
          stem_stride: 2
          final_conv_out_channel: 960
          final_conv_kernel_size: 1
          final_conv_stride: 1
          norm_type: batch_norm
          act_type: relu
          return_stage_idx: ~
          layer_scale: ~
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # FusedIB: ['fi', out_channels, hidden_channels, kernel_size, stride]
          # UniversalInvertedResidualBlock: ['uir', out_channels, hidden_channels, extra_dw, extra_dw_kernel_size, middle_dw, middle_dw_kernel_size, stride]
          # MobileMultiQueryAttention2D: ['mmqa', out_channels, attention_channel, num_attention_heads, query_pooling_stride, key_val_downsample, key_val_downsample_kernel_size, key_val_downsample_stride, stride]
          - 
            - ['fi', 48, 128, 3, 2]
          - 
            - ['uir', 80, 192, True, 3, True, 5, 2]
            - ['uir', 80, 160, True, 3, True, 3, 1]
          - 
            - ['uir', 160, 480, True, 3, True, 5, 2]
            - ['uir', 160, 640, True, 3, True, 3, 1]
            - ['uir', 160, 640, True, 3, True, 3, 1]
            - ['uir', 160, 640, True, 3, True, 5, 1]
            - ['uir', 160, 640, True, 3, True, 3, 1]
            - ['uir', 160, 640, True, 3, False, ~, 1]
            - ['uir', 160, 320, False, ~, False, ~, 1]
            - ['uir', 160, 640, True, 3, False, ~, 1]
          - 
            - ['uir', 256, 960, True, 5, True, 5, 2]
            - ['uir', 256, 1024, True, 5, True, 5, 1]
            - ['uir', 256, 1024, True, 3, True, 5, 1]
            - ['uir', 256, 1024, True, 3, True, 5, 1]
            - ['uir', 256, 1024, False, ~, False, ~, 1]
            - ['uir', 256, 1024, True, 3, False, ~, 1]
            - ['uir', 256, 512, True, 3, True, 5, 1]
            - ['uir', 256, 1024, True, 5, True, 5, 1]
            - ['uir', 256, 1024, False, ~, False, ~, 1]
            - ['uir', 256, 1024, False, ~, False, ~, 1]
            - ['uir', 256, 512, True, 5, False, ~, 1]
  ```
</details>

<details>
  <summary>MobileNetV4-conv-large</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilenetv4
        params:
          stem_out_channel: 24
          stem_kernel_size: 3
          stem_stride: 2
          final_conv_out_channel: 960
          final_conv_kernel_size: 1
          final_conv_stride: 1
          norm_type: batch_norm
          act_type: relu
          return_stage_idx: ~
          layer_scale: ~
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # FusedIB: ['fi', out_channels, hidden_channels, kernel_size, stride]
          # UniversalInvertedResidualBlock: ['uir', out_channels, hidden_channels, extra_dw, extra_dw_kernel_size, middle_dw, middle_dw_kernel_size, stride]
          # MobileMultiQueryAttention2D: ['mmqa', out_channels, attention_channel, num_attention_heads, query_pooling_stride, key_val_downsample, key_val_downsample_kernel_size, key_val_downsample_stride, stride]
          - 
            - ['fi', 48, 96, 3, 2]
          - 
            - ['uir', 96, 192, True, 3, True, 5, 2]
            - ['uir', 96, 384, True, 3, True, 3, 1]
          - 
            - ['uir', 192, 384, True, 3, True, 5, 2]
            - ['uir', 192, 768, True, 3, True, 3, 1]
            - ['uir', 192, 768, True, 3, True, 3, 1]
            - ['uir', 192, 768, True, 3, True, 3, 1]
            - ['uir', 192, 768, True, 3, True, 5, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['uir', 192, 768, True, 3, False, ~, 1]
          - 
            - ['uir', 512, 768, True, 5, True, 5, 2]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, True, 3, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, True, 3, 1]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
  ```
</details>

<details>
  <summary>MobileNetV4-hybrid-medium</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilenetv4
        params:
          stem_out_channel: 32
          stem_kernel_size: 3
          stem_stride: 2
          final_conv_out_channel: 960
          final_conv_kernel_size: 1
          final_conv_stride: 1
          norm_type: batch_norm
          act_type: relu
          return_stage_idx: ~
          layer_scale: 0.1
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # FusedIB: ['fi', out_channels, hidden_channels, kernel_size, stride]
          # UniversalInvertedResidualBlock: ['uir', out_channels, hidden_channels, extra_dw, extra_dw_kernel_size, middle_dw, middle_dw_kernel_size, stride]
          # MobileMultiQueryAttention2D: ['mmqa', out_channels, attention_channel, num_attention_heads, query_pooling_stride, key_val_downsample, key_val_downsample_kernel_size, key_val_downsample_stride, stride]
          - 
            - ['fi', 48, 128, 3, 2]
          - 
            - ['uir', 80, 192, True, 3, True, 5, 2]
            - ['uir', 80, 160, True, 3, True, 3, 1]
          - 
            - ['uir', 160, 480, True, 3, True, 5, 2]
            - ['uir', 160, 320, False, ~, False, ~, 1]
            - ['uir', 160, 640, True, 3, True, 3, 1]
            - ['uir', 160, 640, True, 3, True, 5, 1]
            - ['mmqa', 160, 256, 4, ~, True, 3, 2, 1]
            - ['uir', 160, 640, True, 3, True, 3, 1]
            - ['mmqa', 160, 256, 4, ~, True, 3, 2, 1]
            - ['uir', 160, 640, True, 3, False, ~, 1]
            - ['mmqa', 160, 256, 4, ~, True, 3, 2, 1]
            - ['uir', 160, 640, True, 3, True, 3, 1]
            - ['mmqa', 160, 256, 4, ~, True, 3, 2, 1]
            - ['uir', 160, 640, True, 3, False, ~, 1]
          - 
            - ['uir', 256, 960, True, 5, True, 5, 2]
            - ['uir', 256, 1024, True, 5, True, 5, 1]
            - ['uir', 256, 1024, True, 3, True, 5, 1]
            - ['uir', 256, 1024, True, 3, True, 5, 1]
            - ['uir', 256, 512, False, ~, False, ~, 1]
            - ['uir', 256, 512, True, 3, True, 5, 1]
            - ['uir', 256, 512, False, ~, False, ~, 1]
            - ['uir', 256, 1024, False, ~, False, ~, 1]
            - ['mmqa', 256, 256, 4, ~, False, ~, ~, 1]
            - ['uir', 256, 1024, True, 3, False, ~, 1]
            - ['mmqa', 256, 256, 4, ~, False, ~, ~, 1]
            - ['uir', 256, 1024, True, 5, True, 5, 1]
            - ['mmqa', 256, 256, 4, ~, False, ~, ~, 1]
            - ['uir', 256, 1024, True, 5, False, ~, 1]
            - ['mmqa', 256, 256, 4, ~, False, ~, ~, 1]
            - ['uir', 256, 1024, True, 5, False, ~, 1]
  ```
</details>

<details>
  <summary>MobileNetV4-hybrid-large</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilenetv4
        params:
          stem_out_channel: 24
          stem_kernel_size: 3
          stem_stride: 2
          final_conv_out_channel: 960
          final_conv_kernel_size: 1
          final_conv_stride: 1
          norm_type: batch_norm
          act_type: gelu
          return_stage_idx: ~
          layer_scale: 0.1
        stage_params:
          # Conv2D: ['conv', out_channels, kernel_size, stride]
          # FusedIB: ['fi', out_channels, hidden_channels, kernel_size, stride]
          # UniversalInvertedResidualBlock: ['uir', out_channels, hidden_channels, extra_dw, extra_dw_kernel_size, middle_dw, middle_dw_kernel_size, stride]
          # MobileMultiQueryAttention2D: ['mmqa', out_channels, attention_channel, num_attention_heads, query_pooling_stride, key_val_downsample, key_val_downsample_kernel_size, key_val_downsample_stride, stride]
          - 
            - ['fi', 48, 96, 3, 2]
          - 
            - ['uir', 96, 192, True, 3, True, 5, 2]
            - ['uir', 96, 384, True, 3, True, 3, 1]
          - 
            - ['uir', 192, 384, True, 3, True, 5, 2]
            - ['uir', 192, 768, True, 3, True, 3, 1]
            - ['uir', 192, 768, True, 3, True, 3, 1]
            - ['uir', 192, 768, True, 3, True, 3, 1]
            - ['uir', 192, 768, True, 3, True, 5, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['mmqa', 192, 384, 8, ~, True, 3, 2, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['mmqa', 192, 384, 8, ~, True, 3, 2, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['mmqa', 192, 384, 8, ~, True, 3, 2, 1]
            - ['uir', 192, 768, True, 5, True, 3, 1]
            - ['mmqa', 192, 384, 8, ~, True, 3, 2, 1]
            - ['uir', 192, 768, True, 3, False, ~, 1]
          - 
            - ['uir', 512, 768, True, 5, True, 5, 2]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, True, 3, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['uir', 512, 2048, True, 5, True, 3, 1]
            - ['uir', 512, 2048, True, 5, True, 5, 1]
            - ['mmqa', 512, 512, 8, ~, False, ~, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['mmqa', 512, 512, 8, ~, False, ~, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['mmqa', 512, 512, 8, ~, False, ~, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
            - ['mmqa', 512, 512, 8, ~, False, ~, ~, 1]
            - ['uir', 512, 2048, True, 5, False, ~, 1]
  ```
</details>

## Related links
- [`timm`](https://huggingface.co/blog/rwightman/mobilenetv4)