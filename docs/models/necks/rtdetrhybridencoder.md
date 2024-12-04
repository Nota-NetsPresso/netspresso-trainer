# RT-DETR Hybrid Encoder

RT-DETR Hybrid Encoder based on [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)



## Field lists
| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "rtdetr_hybrid_encoder" to use RT-DETR Hybrid Encoder. |
| `params.hidden_dim` | (int) Hidden dimension size, default is 256 according to paper's Appendix Table A | 
| `params.use_encoder_idx` | (list) Index indicating which feature level to apply encoder. Default is [2] since paper's Section 4.2 mentions AIFI only performed on S5 (highest level) |
| `params.num_encoder_layers` | (int) Number of encoder layers. |
| `params.pe_temperature` | (float) Temperature for positional encoding |
| `params.num_attention_heads` | (int) Number of attention heads. |
| `params.dim_feedforward` | (int) Dimension of feedforward network. |
| `params.dropout` | (float) Dropout rate, default is 0.0 according to configuration |
| `params.attn_act_type` | (str) Activation function type for attention, using GELU |
| `params.expansion` | (float) Expansion ratio for RepBlock in CCFF module, default is 0.5 |
| `params.depth_mult` | (float) Depth multiplier for scaling. |
| `params.conv_act_type` | (str) Activation function type for convolution layers, using SiLU according to paper's Figure 4. |


## Model configuration examples

<details>
  <summary>RT-DETR Hybrid Encoder</summary>
  
  ```yaml
  model:
    architecture:
      neck:
        name: fpn
        params:
          num_outs: 4
          start_level: 0
          end_level: -1
          add_extra_convs: False
          relu_before_extra_convs: False
  ```
</details>

## Related links

