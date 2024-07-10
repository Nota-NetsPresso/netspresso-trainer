# MobileViT

MobileViT backbone based on [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178).

MobileViT was introduced by combining inverted residual blocks with transformer-based MobileViT blocks. In line with this, it is possible to select between inverted residual blocks (as mv2) and MobileViT models for each stage of the backbone with detailed configurations according to each block type.

## Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "mobilevit" to use MobileViT backbone. |
| `params.patch_size` | (int) Patch size for MobileViT blocks. |
| `params.num_attention_heads` | (int) The number of heads in the multi-head attention. |
| `params.attention_dropout_prob` | (float) Dropout probability in the attention. |
| `params.ffn_dropout_prob` | (float) Dropout probability in the feed-forward network inside of the attention block. |
| `params.output_expansion_ratio` | (int) Expansion ratio for computing output dimension of the model. If expanded dimension is bigger than 960, it is set to 960. |
| `params.use_fusion_layer` | (bool) Whether to use fusion layer for MobileViT blocks. |
| `stage_params[n].block_type` | (str) Determines which block to use, "mv2" or "mobilevit". |
| `stage_params[n].out_channels` | (int) Output dimension of the block. |
| `stage_params[n].num_blocks` | (int) The number of blocks in the stage. Note that if `block_type` is `mobilevit`, an extra inverted residual block is added before MobileViT blocks. |
| `stage_params[n].stride` | (int) Stride value for the block. |
| `stage_params[n].attention_channels` | (int) Dimension for the attention block. If is used only `block_type` is "mobilevit". |
| `stage_params[n].ffn_intermediate_channels` | (int) Intermediate dimension for the feed forward network inside of the attention block. |
| `stage_params[n].dilate` | (bool) Whether to replace stride as dilated convolution. It is used only `block_type` is `mobilevit`. |
| `stage_params[n].ir_expansion_ratio` | (int) Dimension expansion ratio for inverted residual block. |

## Model configuration examples

<details>
  <summary>MobileViT-xxs</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilevit
        params:
          patch_size: 2
          num_attention_heads: 4  # num_heads
          attention_dropout_prob: 0.1
          ffn_dropout_prob: 0.0
          output_expansion_ratio: 4
          use_fusion_layer: True
        stage_params:
          -
            block_type: 'mv2'
            out_channels: 16
            num_blocks: 1
            stride: 1
            ir_expansion_ratio: 2
          -
            block_type: 'mv2'
            out_channels: 24
            num_blocks: 3
            stride: 2
            ir_expansion_ratio: 2
          -
            block_type: 'mobilevit'
            out_channels: 48
            num_blocks: 2
            stride: 2
            attention_channels: 64
            ffn_intermediate_channels: 128
            dilate: False
            ir_expansion_ratio: 2
          -
            block_type: 'mobilevit'
            out_channels: 64
            num_blocks: 4
            stride: 2
            attention_channels: 80
            ffn_intermediate_channels: 160
            dilate: False
            ir_expansion_ratio: 2
          -
            block_type: 'mobilevit'
            out_channels: 80
            num_blocks: 3
            stride: 2
            attention_channels: 96
            ffn_intermediate_channels: 192
            dilate: False
            ir_expansion_ratio: 2
  ```
</details>


<details>
  <summary>MobileViT-xs</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilevit
        params:
          patch_size: 2
          num_attention_heads: 4  # num_heads
          attention_dropout_prob: 0.1
          ffn_dropout_prob: 0.0
          output_expansion_ratio: 4
          use_fusion_layer: True
        stage_params:
          -
            block_type: 'mv2'
            out_channels: 32
            num_blocks: 1
            stride: 1
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mv2'
            out_channels: 48
            num_blocks: 3
            stride: 2
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mobilevit'
            out_channels: 64
            num_blocks: 2
            stride: 2
            attention_channels: 96
            ffn_intermediate_channels: 192
            dilate: False
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mobilevit'
            out_channels: 80
            num_blocks: 4
            stride: 2
            attention_channels: 120
            ffn_intermediate_channels: 240
            dilate: False
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mobilevit'
            out_channels: 96
            num_blocks: 3
            stride: 2
            attention_channels: 144
            ffn_intermediate_channels: 288
            dilate: False
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
  ```
</details>

<details>
  <summary>MobileViT-s</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mobilevit
        params:
          patch_size: 2
          num_attention_heads: 4  # num_heads
          attention_dropout_prob: 0.1
          ffn_dropout_prob: 0.0
          output_expansion_ratio: 4
          use_fusion_layer: True
        stage_params:
          -
            block_type: 'mv2'
            out_channels: 32
            num_blocks: 1
            stride: 1
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mv2'
            out_channels: 64
            num_blocks: 3
            stride: 2
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mobilevit'
            out_channels: 96
            num_blocks: 2
            stride: 2
            attention_channels: 144
            ffn_intermediate_channels: 288
            dilate: False
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mobilevit'
            out_channels: 128
            num_blocks: 4
            stride: 2
            attention_channels: 192
            ffn_intermediate_channels: 384
            dilate: False
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
          -
            block_type: 'mobilevit'
            out_channels: 160
            num_blocks: 3
            stride: 2
            attention_channels: 240
            ffn_intermediate_channels: 480
            dilate: False
            ir_expansion_ratio: 4  # [mv2_exp_mult] * 4
  ```
</details>

## Related links
- [`apple/ml-cvnets`](https://github.com/apple/ml-cvnets/tree/cvnets-v0.2)
