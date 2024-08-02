# ResNet

ResNet backbone based on [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).

You can flexibly choose between basicblock and bottleneck as the building blocks for the ResNet architecture. And, you can also freely determine the number of stages and the repetition of blocks within the model. This flexibility supports the creation of various ResNet models, e.g. ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152. Also this supports adjusting the number of stages and blocks for your specific requirements.

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "resnet" to use `ResNet` backbone. |
| `params.block_type` | (str) Key value that determines which block to use, "basicblock" or "bottleneck". |
| `params.norm_type` | (str) Type of normalization layer. Supporting normalization layers are described in [[here]](../layers/normalizations.md). |
| `stage_params[n].channels` | (int) The dimension of the first convolution layer in each block. |
| `stage_params[n].num_blocks` | (int) The number of blocks in the stage. |
| `stage_params[n].replace_stride_with_dilation` | (bool) Flag that determines whether to replace stride step with dilated convolution. |

## Model configuration examples

<details>
  <summary>ResNet18</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: resnet
        params:
          block: basicblock
          norm_layer: batch_norm
        stage_params:
          - 
            channels: 64
            layers: 2
          - 
            channels: 128
            layers: 2
            replace_stride_with_dilation: False
          - 
            channels: 256
            layers: 2
            replace_stride_with_dilation: False
          - 
            plane: 512
            layers: 2
            replace_stride_with_dilation: False
  ```
</details>

<details>
  <summary>ResNet34</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: resnet
        params:
          block: basicblock
          norm_layer: batch_norm
        stage_params:
          - 
            plane: 64
            layers: 3
          - 
            plane: 128
            layers: 4
            replace_stride_with_dilation: False
          - 
            plane: 256
            layers: 6
            replace_stride_with_dilation: False
          - 
            plane: 512
            layers: 3
            replace_stride_with_dilation: False
  ```
</details>

<details>
  <summary>ResNet50</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: resnet
        params:
          block: bottleneck
          norm_layer: batch_norm
        stage_params:
          - 
            plane: 64
            layers: 3
          - 
            plane: 128
            layers: 4
            replace_stride_with_dilation: False
          - 
            plane: 256
            layers: 6
            replace_stride_with_dilation: False
          - 
            plane: 512
            layers: 3
            replace_stride_with_dilation: False
  ```
</details>


<details>
  <summary>ResNet101</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: resnet
        params:
          block: bottleneck
          norm_layer: batch_norm
        stage_params:
          - 
            plane: 64
            layers: 3
          - 
            plane: 128
            layers: 4
            replace_stride_with_dilation: False
          - 
            plane: 256
            layers: 23
            replace_stride_with_dilation: False
          - 
            plane: 512
            layers: 3
            replace_stride_with_dilation: False
  ```
</details>

<details>
  <summary>ResNet152</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: resnet
        params:
          block: bottleneck
          norm_layer: batch_norm
        stage_params:
          - 
            plane: 64
            layers: 3
          - 
            plane: 128
            layers: 8
            replace_stride_with_dilation: False
          - 
            plane: 256
            layers: 36
            replace_stride_with_dilation: False
          - 
            plane: 512
            layers: 3
            replace_stride_with_dilation: False
  ```
</details>

## Related links
- [`pytorch/vision`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
