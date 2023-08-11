# Models

> :bulb: Note that all FLOPs and # Params values in each task section includes backbones, not only head itself.
## ResNet

- Original Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Related Links
    - [`pytorch/vision`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

<table>
  <tr>
    <th>Model</th>
    <th>Supporting Task(s)</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
    <th>Checkpoint</th>
  </tr>
  <tr>
    <td><code>resnet50</code></td>
    <td>Classification</td>
    <td>✅</td>
    <td>✅</td>
    <td><a href="https://drive.google.com/file/d/1xFfPcea8VyZ5KlegrIcSMUpRZ-FKOvKF/view?usp=drive_link">Google Drive</a></td>
  </tr>
</table>

### Classification

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>fc</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## ViT

- Original Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- Related Links
    - [`apple/ml-cvnets`](https://github.com/apple/ml-cvnets/tree/cvnets-v0.1)

<table>
  <tr>
    <th>Model</th>
    <th>Supporting Task(s)</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
    <th>Checkpoint</th>
  </tr>
  <tr>
    <td><code>vit_tiny</code></td>
    <td>Classification</td>
    <td>✅</td>
    <td>✅</td>
    <td><a href="https://drive.google.com/file/d/1meGp4epqXcqplHnSkXHIVuvV2LYSaLFU/view?usp=drive_link">Google Drive</a></td>
  </tr>
</table>

### Classification

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>fc</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## MobileViT

- Original Paper: [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)
- Related Links
    - [`apple/ml-cvnets`](https://github.com/apple/ml-cvnets/tree/cvnets-v0.2)

<table>
  <tr>
    <th>Model</th>
    <th>Supporting Task(s)</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
    <th>Checkpoint</th>
  </tr>
  <tr>
    <td><code>mobilevit_s</code></td>
    <td>Classification</td>
    <td>✅</td>
    <td>✅</td>
    <td><a href="https://drive.google.com/file/d/1HF6iq1T0QSUqPViJobXx639xlBxkBHWd/view?usp=drive_link">Google Drive</a></td>
  </tr>
</table>

### Classification

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>fc</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## SegFormer

- Original Paper: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- Related Links
    - [`huggingface/transformers`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer)

<table>
  <tr>
    <th>Model</th>
    <th>Supporting Task(s)</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
    <th>Checkpoint</th>
  </tr>
  <tr>
    <td><code>segformer</code></td>
    <td>Classification,<br />Segmentation</td>
    <td>✅</td>
    <td>✅</td>
    <td><a href="https://drive.google.com/file/d/1QIvgBOwGKXfUS9ysDk3K9AkTAOaiyRXK/view?usp=drive_link">Google Drive</a></td>
  </tr>
</table>

### Classification

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>fc</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

### Segmentation

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>decode_head</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## EfficientFormer

- Original Paper: [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)
- Related Links
    - [`snap-research/EfficientFormer`](https://github.com/snap-research/EfficientFormer)

<table>
  <tr>
    <th>Model</th>
    <th>Supporting Task(s)</th>
    <th>torch.fx</th>
    <th>NetsPresso</th>
    <th>Checkpoint</th>
  </tr>
  <tr>
    <td><code>segformer</code></td>
    <td>Classification,<br />Segmentation</td>
    <td>✅</td>
    <td>✅</td>
    <td><a href="https://drive.google.com/file/d/1I0SoTFs5AcI3mHpG_kDM2mW1PXDmG8X_/view?usp=drive_link">Google Drive</a></td>
  </tr>
</table>

### Classification

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>fc</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

### Segmentation

<table>
  <tr>
    <th>Head</th>
    <th>FLOPs</th>
    <th># Params</th>
    <th>Speed</th>
    <th>Benchmark result(s)</th>
  </tr>
  <tr>
    <td><code>decode_head</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>