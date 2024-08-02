# FC

Fully connected layer head for classification. You can adjust the number of layers and channel sizes. Channel of last layer is always same with the number of classes.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "fc" to use `FC` head. |
| `params.num_layers` | (int) The number of fully connected layers. Channel of last layer is same with the number of classes. |
| `params.intermediate_channels` | (int) Dimension of intermediate fully connected layers. This can be ignored if `num_layer` is 1. |
| `params.act_type` | (str) Activation function for intermediate fully connected layers. This can be ignored if `num_layer` is 1. |
| `params.dropout_prob` | (float) Dropout probability before the last classifier layer. |

## Model configuration example

<details>
  <summary>2-layer fully connected layer classifier</summary>
  
  ```yaml
  model:
    architecture:
      head:
        name: fc
        params:
          num_layers: 2
          intermediate_channels: 1024
          act_type: hard_swish
          dropout_prob: 0.2
  ```
</details>

## Related links