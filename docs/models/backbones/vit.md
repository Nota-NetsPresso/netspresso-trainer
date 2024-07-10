# ViT

ViT backbone based on [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

ViT (Vision Transformer) does not have a stage configuration and therefore does not support compatibility with neck modules. Currently, it only supports the FC head. When using the ViT model for classification tasks, users can decide whether to use a classification token. Additionally, users can flexibly configure the settings of the transformer encoder.

## Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "vit" to use `ViT` backbone. |
| `params.patch_size` | (int) Size of the image patch to be treated as a single embedding. |
| `params.attention_channels` | (int) Dimension for the encoder. |
| `params.num_blocks` | (int) The number of self-attention blocks in the encoder. |
| `params.num_attention_heads` | (int) The number of heads in the multi-head attention. |
| `params.attention_dropout_prob` | (float) Dropout probability in the attention block. |
| `params.ffn_intermediate_channels` | (int) Intermediate dimension of the feed-forward network inside the attention block. |
| `params.ffn_dropout_prob` | (float) Dropout probability in the feed-forward network inside the attention block. |
| `params.use_cls_token` | (bool) Whether to use the classification token. |
| `params.vocab_size` | (int) Maximum token length for positional encoding. |

## Model configuration examples

<details>
  <summary>ViT-tiny</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: vit
        params:
          patch_size: 16
          attention_channels: 192
          num_blocks: 12
          num_attention_heads: 3
          attention_dropout_prob: 0.0
          ffn_intermediate_channels: 768  # hidden_size * 4
          ffn_dropout_prob: 0.1
          use_cls_token: True
          vocab_size: 1000
        stage_params: ~
  ```
</details>

<details>
  <summary>ViT-small</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: vit
        params:
          patch_size: 16
          attention_channels: 384
          num_blocks: 12
          num_attention_heads: 6
          attention_dropout_prob: 0.0
          ffn_intermediate_channels: 1536  # hidden_size * 4
          ffn_dropout_prob: 0.0
          use_cls_token: True
          vocab_size: 1000
        stage_params: ~
  ```
</details>



## Related links
- [`apple/ml-cvnets`](https://github.com/apple/ml-cvnets/tree/cvnets-v0.1)
