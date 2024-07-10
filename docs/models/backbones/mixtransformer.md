# MixTransformer

MixTransformer backbone based on [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf).

We provide the MixTransformer encoder (MiT), the backbone of SegFormer, as a freely usable backbone module. Users have the flexibility to configure the transformer encoder for each stage, enabling MiT-b0 to MiT-b5.

## Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "mixtransformer" to use `MixTransformer` backbone. |
| `params.ffn_intermediate_expansion_ratio` | (int) Expansion factor to compute intermediate dimension in feed-forward network. |
| `params.ffn_act_type` | (str) Activation function for feed-forward network in the transformer block. Supporting activation functions are described in [[here]](../../components/model/activations.md). |
| `params.ffn_dropout_prob` | (float) Dropout probability for feed-forward network in the transformer block. |
| `params.attention_dropout_prob` | (float) Dropout probability for attention in the transformer block. |
| `stage_params[n].num_blocks` | (int) The number of transformer blocks in the encoder. |
| `stage_params[n].sequence_reduction_ratio` | (int) Sequence reduction ratio for multi-head attention. |
| `stage_params[n].encoder_chananels` | (int) Dimension for the transformer block. |
| `stage_params[n].embedding_patch_sizes` | (int) Kernel size for convolution layer in overlapping patch embedding. |
| `stage_params[n].embedding_strides` | (int) stride value for convolution layer in overlapping patch embedding. |
| `stage_params[n].num_attention_heads` | (int) The number of heads in the multi-head attention. |

## Model configuration examples

<details>
  <summary>MiT-b0</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mixtransformer
        params:
          ffn_intermediate_expansion_ratio: 4
          ffn_act_type: "gelu"
          ffn_dropout_prob: 0.0
          attention_dropout_prob: 0.0
        stage_params:
          -
            num_blocks: 2
            sequence_reduction_ratio: 8
            attention_chananels: 32
            embedding_patch_sizes: 7
            embedding_strides: 4
            num_attention_heads: 1
          -
            num_blocks: 2
            sequence_reduction_ratio: 4
            attention_chananels: 64
            embedding_patch_sizes: 3
            embedding_strides: 2
            num_attention_heads: 2
          -
            num_blocks: 2
            sequence_reduction_ratio: 2
            attention_chananels: 160
            embedding_patch_sizes: 3
            embedding_strides: 2
            num_attention_heads: 5
          -
            num_blocks: 2
            sequence_reduction_ratio: 1
            attention_chananels: 256
            embedding_patch_sizes: 3
            embedding_strides: 2
            num_attention_heads: 8
  ```
</details>

## Related links
- [`huggingface/transformers`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer)
