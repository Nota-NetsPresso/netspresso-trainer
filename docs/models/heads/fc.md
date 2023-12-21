# FC

Fully connected layer head for classification. You can adjust the number of layers and channel sizes. Channel of last layer is always same with the number of classes.

## Compatibility matrix

<table>
  <tr>
    <th>Supporting backbones</th>
    <th>Supporting necks</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
  </tr>
  <tr>
    <td>
      ResNet<br />
      MobileNetV3<br />
      MixNet<br />
      CSPDarkNet<br />
      ViT<br />
      MobileViT<br />
      MixTransformer<br />
      EfficientFormer
    </td>
    <td>
    </td>
    <td>Supported</td>
    <td>Supported</td>
  </tr>
</table>


## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "fc" to use `FC` head. |
| `params.intermediate_channels` | (int) Dimension of intermediate fully connected layers. |
| `params.num_layers` | (int) The number of fully connected layers. Channel of last layer is same with the number of classes. |

## Model configuration example

<details>
  <summary>2-layer fully connected layer classifier</summary>
  
  ```yaml
  model:
    architecture:
      head:
        name: fc
        params:
          hidden_size: 1024
          num_layers: 2
  ```
</details>

## Related links