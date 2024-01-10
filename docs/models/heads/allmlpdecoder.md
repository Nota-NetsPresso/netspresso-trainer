# AllMLPDecoder

All-MLP decoder based on [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf).

We provide the AllMLP Decoder, the head of SegFormer, as a freely usable head module. AllMLP Decoder takes intermediate features from previous backbone or neck module and outputs a segmentation map of the target size.

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
      MobileViT<br />
      MixTransformer<br />
      EfficientFormer
    </td>
    <td>
    FPN<br />
    YOLOPAFPN<br />
    </td>
    <td>Supported</td>
    <td>Supported</td>
  </tr>
</table>


## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "all_mlp_decoder" to use `AllMLPDecoder` head. |
| `params.intermediate_channels` | (int) Intermediate feature dimension of the decoder. |
| `params.classifier_dropout_prob` | (float) Dropout probability of classifier. |

## Model configuration example

<details>
  <summary>AllMLP decoder</summary>
  
  ```yaml
  model:
    architecture:
      head:
        name: all_mlp_decoder
        params:
          intermediate_channels: 256
          classifier_dropout_prob: 0.
  ```
</details>

## Related links