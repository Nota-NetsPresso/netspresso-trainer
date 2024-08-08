# MixNet

MixNet backbone based on [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/pdf/1907.09595v3.pdf).

Similar with MobileNetV3, we provide a configuration that allows users to define each MixDepthBlock in MixNet individually. You can specify in a list format the number and form of each MixDepthBlock for every stage. Using this, we pre-construct and provide MixNet family models.

## Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "mixnet" to use `MixNet` backbone. |
| `params.stem_channels` | (int) Output dimension of the first convolution layer. |
| `params.wid_mul` | (float) Ratio for adjusting the input/output dimensions for the entire model. |
| `params.dep_mul` | (float) Ratio for adjusting the `num_block` value for the entire model. |
| `params.dropout_rate` | (float) Dropout ratio applied to all `MixDepthBlock`. |
| `stage_params[n].expansion_ratio` | (list[int]) Determines the output dimension of the expansion phase in each `MixDepthBlock`. Expands the input dimensions by multiplying the `expansion_ratio`. |
| `stage_params[n].out_channels` | (list[int]) Output dimensions of each `MixDepthBlock`. |
| `stage_params[n].num_blocks` | (list[int]) Repetition count for each `MixDepthBlock`. |
| `stage_params[n].kernel_sizes` | (list[list[int]]) Various kernel sizes used within each `MixDepthBlock`. |
| `stage_params[n].num_exp_groups` | (list[int]) The number of convolution groups in the expansion phase of `MixDepthBlock`. |
| `stage_params[n].num_poi_groups` | (list[int]) The number of convolution groups in the final point-wise convolution of `MixDepthBlock`. |
| `stage_params[n].stride` | (list[int]) Stride values for each `MixDepthBlock`. |
| `stage_params[n].act_type` | (list[str]) Activation function for each `MixDepthBlock`. Supporting activation functions are described in [[here]](../../components/model/activations.md) |
| `stage_params[n].se_reduction_ratio` | (list[int]) Reduction factor for calculating the output dimension of the squeeze-and-excitation block. If `None`, the squeeze-and-excitation block is not applied. |

## Model configuration examples

<details>
  <summary>MixNet-s</summary>

  ```yaml
  model:
    architecture:
      backbone:
        name: mixnet
        params:
          stem_channels: 16
          wid_mul: 1.0
          dep_mul: 1.0
          dropout_rate: 0.
        stage_params: 
          -
            expansion_ratio: [1, 6, 3]
            out_channels: [16, 24, 24]
            num_blocks: [1, 1, 1]
            kernel_sizes: [[3], [3], [3]]
            num_exp_groups: [1, 2, 2]
            num_poi_groups: [1, 2, 2]
            stride: [1, 2, 1]
            act_type: ["relu", "relu", "relu"]
            se_reduction_ratio: [~, ~, ~]
          -
            expansion_ratio: [6, 6]
            out_channels: [40, 40]
            num_blocks: [1, 3]
            kernel_sizes: [[3, 5, 7], [3, 5]]
            num_exp_groups: [1, 2]
            num_poi_groups: [1, 2]
            stride: [2, 1]
            act_type: ["swish", "swish"]
            se_reduction_ratio: [2, 2]
          -
            expansion_ratio: [6, 6, 6, 3]
            out_channels: [80, 80, 120, 120]
            num_blocks: [1, 2, 1, 2]
            kernel_sizes: [[3, 5, 7], [3, 5], [3, 5, 7], [3, 5, 7, 9]]
            num_exp_groups: [1, 1, 2, 2]
            num_poi_groups: [2, 2, 2, 2]
            stride: [2, 1, 1, 1]
            act_type: ["swish", "swish", "swish", "swish"]
            se_reduction_ratio: [4, 4, 2, 2]
          -
            expansion_ratio: [6, 6]
            out_channels: [200, 200]
            num_blocks: [1, 2]
            kernel_sizes: [[3, 5, 7, 9, 11], [3, 5, 7, 9]]
            num_exp_groups: [1, 1]
            num_poi_groups: [1, 2]
            stride: [2, 1]
            act_type: ["swish", "swish"]
            se_reduction_ratio: [2, 2]
  ```
</details>

<details>
  <summary>MixNet-m</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mixnet
        params:
          stem_channels: 24
          wid_mul: 1.0
          dep_mul: 1.0
          dropout_rate: 0.
        stage_params: 
          -
            expansion_ratio: [1, 6, 3]
            out_channels: [24, 32, 32]
            num_blocks: [1, 1, 1]
            kernel_sizes: [[3], [3, 5, 7], [3]]
            num_exp_groups: [1, 2, 2]
            num_poi_groups: [1, 2, 2]
            stride: [1, 2, 1]
            act_type: ["relu", "relu", "relu"]
            se_reduction_ratio: [~, ~, ~]
          -
            expansion_ratio: [6, 6]
            out_channels: [40, 40]
            num_blocks: [1, 3]
            kernel_sizes: [[3, 5, 7, 9], [3, 5]]
            num_exp_groups: [1, 2]
            num_poi_groups: [1, 2]
            stride: [2, 1]
            act_type: ["swish", "swish"]
            se_reduction_ratio: [2, 2]
          -
            expansion_ratio: [6, 6, 6, 3]
            out_channels: [80, 80, 120, 120]
            num_blocks: [1, 3, 1, 3]
            kernel_sizes: [[3, 5, 7], [3, 5, 7, 9], [3], [3, 5, 7, 9]]
            num_exp_groups: [1, 2, 1, 2]
            num_poi_groups: [1, 2, 1, 2]
            stride: [2, 1, 1, 1]
            act_type: ["swish", "swish", "swish", "swish"]
            se_reduction_ratio: [4, 4, 2, 2]
          -
            expansion_ratio: [6, 6]
            out_channels: [200, 200]
            num_blocks: [1, 3]
            kernel_sizes: [[3, 5, 7, 9], [3, 5, 7, 9]]
            num_exp_groups: [1, 1]
            num_poi_groups: [1, 2]
            stride: [2, 1]
            act_type: ["swish", "swish"]
            se_reduction_ratio: [2, 2]
  ```
</details>

<details>
  <summary>MixNet-l</summary>
  
  ```yaml
  model:
    architecture:
      backbone:
        name: mixnet
        params:
          stem_channels: 24
          wid_mul: 1.3
          dep_mul: 1.0
          dropout_rate: 0.
        stage_params: 
          -
            expansion_ratio: [1, 6, 3]
            out_channels: [24, 32, 32]
            num_blocks: [1, 1, 1]
            kernel_sizes: [[3], [3, 5, 7], [3]]
            num_exp_groups: [1, 2, 2]
            num_poi_groups: [1, 2, 2]
            stride: [1, 2, 1]
            act_type: ["relu", "relu", "relu"]
            se_reduction_ratio: [~, ~, ~]
          -
            expansion_ratio: [6, 6]
            out_channels: [40, 40]
            num_blocks: [1, 3]
            kernel_sizes: [[3, 5, 7, 9], [3, 5]]
            num_exp_groups: [1, 2]
            num_poi_groups: [1, 2]
            stride: [2, 1]
            act_type: ["swish", "swish"]
            se_reduction_ratio: [2, 2]
          -
            expansion_ratio: [6, 6, 6, 3]
            out_channels: [80, 80, 120, 120]
            num_blocks: [1, 3, 1, 3]
            kernel_sizes: [[3, 5, 7], [3, 5, 7, 9], [3], [3, 5, 7, 9]]
            num_exp_groups: [1, 2, 1, 2]
            num_poi_groups: [1, 2, 1, 2]
            stride: [2, 1, 1, 1]
            act_type: ["swish", "swish", "swish", "swish"]
            se_reduction_ratio: [4, 4, 2, 2]
          -
            expansion_ratio: [6, 6]
            out_channels: [200, 200]
            num_blocks: [1, 3]
            kernel_sizes: [[3, 5, 7, 9], [3, 5, 7, 9]]
            num_exp_groups: [1, 1]
            num_poi_groups: [1, 2]
            stride: [2, 1]
            act_type: ["swish", "swish"]
            se_reduction_ratio: [2, 2]
  ```
</details>

## Related links