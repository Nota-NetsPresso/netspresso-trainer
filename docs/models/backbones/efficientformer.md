# EfficientFormer

EfficientFormer backbone based on [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191).

EfficientFormer is designed following the design principle of MetaFormer, constructing its backbone by stacking MetaBlocks. 4D MetaBlocks are employed throughout the model, and 3D MetaBlocks are used at the end of the backbone to enhance the model's expression power. We provide configuration options to adjust the design settings including repetition values for 3D and 4D MetaBlocks.

## Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "efficientformer" to use EfficientFormer backbone. |
| `params.num_attention_heads` | (int) The number of heads in the multi-head attention of 3D MetaBlock. |
| `params.attention_channels` | (int) Dimension for attention of 3D MetaBlock. |
| `params.attention_dropout_prob` | (float) Dropout probability for attention of 3D MetaBlock. |
| `params.attention_value_expansion_ratio` | (int) Value dimension expansion ratio of 3D MetaBlock. |
| `params.ffn_intermediate_ratio` | (int) Dimension expansion ratio of MLP layer in 3D and 4D MetaBlock. |
| `params.ffn_dropout_prob` | (float) Dropout probability of MLP layer in 3D and 4D MetaBlock. |
| `params.ffn_act_type` | (str) Activation function of MLP layer in 3D and 4D MetaBlocks |
| `params.vit_num` | (int) The number of last 3D MetaBlock. |
| `stage_params[n].num_blocks` | (int) The number of 4D MetaBlock in the stage. |
| `stage_params[n].channels` | (int) Dimensions for 4D MetaBlock in the stage. |

## Model configuration examples

<details>
  <summary>EfficientFormer-L1</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: efficientformer
        params:
          num_attention_heads: 8
          attention_channels: 256  # attention_hidden_size_splitted * num_attention_heads
          attention_dropout_prob: 0.
          attention_value_expansion_ratio: 4
          ffn_intermediate_ratio: 4
          ffn_dropout_prob: 0.
          ffn_act_type: 'gelu'
          vit_num: 1
        stage_params:
          - 
            num_blocks: 3
            channels: 48
          - 
            num_blocks: 2
            channels: 96
          - 
            num_blocks: 6
            channels: 224
          - 
            num_blocks: 4
            channels: 448
  ```
</details>

<details>
  <summary>EfficientFormer-L3</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: efficientformer
        params:
          num_attention_heads: 8
          attention_channels: 256  # attention_hidden_size_splitted * num_attention_heads
          attention_dropout_prob: 0.
          attention_value_expansion_ratio: 4
          ffn_intermediate_ratio: 4
          ffn_dropout_prob: 0.
          ffn_act_type: 'gelu'
          vit_num: 4
        stage_params:
          - 
            num_blocks: 4
            channels: 64
          - 
            num_blocks: 4
            channels: 128
          - 
            num_blocks: 12
            channels: 320
          - 
            num_blocks: 6
            channels: 512
  ```
</details>

## Related links
- [`snap-research/EfficientFormer`](https://github.com/snap-research/EfficientFormer)
