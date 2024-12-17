# RT-DETR Head
RT-DETR detection head based on [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069).

We provide the head of RT-DETR as `rtdetr_head`. 

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "rtdetr_head" to use `RT-DETR Head` head. |
| `params.hidden_dim` | (int) Hidden dimension size, default is 256 according to paper's Appendix Table A |
| `params.num_attention_heads` | (int) Number of attention heads, default is 8 according to paper's Appendix Table A |
| `params.num_levels` | (int) Number of feature levels used, default is 3 according to paper's Section 4.1 |
| `params.num_queries` | (int) Number of object queries, default is 300 according to paper's Section 4.1 and Appendix Table A |
| `params.eps` | (float) Small constant for numerical stability, default is 1e-2 |
| `params.num_decoder_layers` | (int) Number of decoder layers. |
| `params.position_embed_type` | (str) Type of position embedding used ['sine', 'learned']. |
| `params.num_decoder_points` | (int) Number of decoder reference points, default is 4 according to paper's Appendix Table A. |
| `params.dim_feedforward` | (int) Feedforward network dimension, default is 1024 according to paper's Appendix Table A. |
| `params.dropout` | (float) Dropout rate in layers. |
| `params.act_type` | (str) Activation function type. |
| `params.num_denoising` | (int) Number of denoising queries. |
| `params.label_noise_ratio` | (float) Label noise ratio for denoising training, default is 0.5 according to paper's Appendix Table A. |
| `params.use_aux_loss` | (bool) Whether to use auxiliary loss when training. The paper mentions using auxiliary prediction heads in Section 4.1. |

## Model configuration example

<details>
  <summary>RT-DETR head</summary>
  
  ```yaml
  model:
    architecture:
      head:
        name: rtdetr_head
      params:
        hidden_dim: 256
        num_attention_heads: 8
        num_levels: 3
        num_queries: 300
        eps: 1e-2
        num_decoder_layers: 3
        eval_spatial_size: ~
        position_embed_type: sine
        num_decoder_points: 4
        dim_feedforward: 1024
        dropout: 0.0
        act_type: relu
        num_denoising: 100
        label_noise_ratio: 0.5
        use_aux_loss: true
  ```
</details>

## Related links

- [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) 
- [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)