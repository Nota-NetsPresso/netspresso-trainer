# Backbones

*Only params and stage_params will be contained*

## ResNet

ResNet backbone based on [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

### Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "resnet" to use ResNet backbone. |
| `params.block` | (str) |
| `params.norm_layer` | (str)|
| `params.groups` | (int) |
| `params.width_per_group` | (int) |
| `params.zero_init_residual` | (bool) |
| `params.expansion` | () |
| `stage_params[n].plane` | (int) |
| `stage_params[n].layers` | (int) |
| `stage_params[n].replace_stride_with_dilation` | (bool) |


## MobileNetV3

MobileNetV3 backbone based on [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).

### Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "mobilenetv3" to use MobileNetV3 backbone. |
| `stage_params[n].in_channels` | (list[int]) |
| `stage_params[n].kernel` | (list[int]) |
| `stage_params[n].expanded_channels` | (list[int]) |
| `stage_params[n].out_channels` | (list[int]) |
| `stage_params[n].use_se` | (list[bool]) |
| `stage_params[n].activation` | (list[str]) |
| `stage_params[n].stride` | (list[int]) |
| `stage_params[n].dilation` | (list[int]) |

## MixNet

MixNet backbone based on [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/pdf/1907.09595v3.pdf).

### Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "mixnet" to use MobileNetV3 backbone. |
| `params.stem_planes` | (int) |
| `params.width_multi` | (float) |
| `params.depth_multi` | (float) |
| `params.dropout_rate` | (float) |
| `stage_params[n].expand_ratio` | (list[int]) |
| `stage_params[n].out_channels` | (list[int]) |
| `stage_params[n].num_blocks` | (list[int]) |
| `stage_params[n].kernel_sizes` | (list[list[int]]) |
| `stage_params[n].exp_kernel_sizes` | (list[list[int]]) |
| `stage_params[n].poi_kernel_sizes` | (list[list[int]]) |
| `stage_params[n].stride` | (list[int]) |
| `stage_params[n].dilation` | (list[int]) |
| `stage_params[n].act_type` | (list[str]) |
| `stage_params[n].se_reduction_ratio` | (list[int]) |

## CSPDarkNet

CSPDarkNet backbone based on .

### Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "cspdarknet" to use CSPDarkNet backbone. |
| `params.dep_mul` | (float) |
| `params.wid_mul` | (float) |
| `params.act_type.` | (str) |

## ViT

ViT backbone based on .

### Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "vit" to use ViT backbone. |
| `params.patch_size` | (int) |
| `params.hidden_size` | (int) |
| `params.num_blocks` | (int) |
| `params.num_attention_heads` | (int) |
| `params.attention_dropout_prob` | (float) |
| `params.intermediate_size` | (int) |
| `params.hidden_dropout_prob` | (float) |
| `params.layer_norm_eps` | (float) |
| `params.use_cls_token` | (bool) |
| `params.vocab_size` | (int) |

## MobileViT

MobileViT backbone based on .

### Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "vit" to use ViT backbone. |
| `params.patch_embedding_out_channels` | () |
| `params.local_kernel_size` | () |
| `params.patch_size` | () |
| `params.num_attention_heads` | () |
| `params.attention_dropout_prob` | () |
| `params.hidden_dropout_prob` | () |
| `params.exp_factor` | () |
| `params.layer_norm_eps` | () |
| `params.use_fusion_layer` | () |
| `stage_params[n].out_channels` | () |
| `stage_params[n].block_type` | () |
| `stage_params[n].num_blocks` | () |
| `stage_params[n].stride` | () |
| `stage_params[n].hidden_size` | () |
| `stage_params[n].intermediate_size` | () |
| `stage_params[n].num_transformer_blocks` | () |
| `stage_params[n].dilate` | () |
| `stage_params[n].expand_ratio` | () |

## SegFormer

SegFormer backbone based on .

### Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "vit" to use ViT backbone. |
| `params.intermediate_ratio` | () |
| `params.hidden_activation_type` | () |
| `params.hidden_dropout_prob` | () |
| `params.attention_dropout_prob` | () |
| `params.layer_norm_eps` | () |
| `stage_params[n].num_blocks` | () |
| `stage_params[n].sr_ratios` | () |
| `stage_params[n].hidden_sizes` | () |
| `stage_params[n].embedding_patch_sizes` | () |
| `stage_params[n].embedding_strides` | () |
| `stage_params[n].num_attention_heads` | () |

## EfficientFormer

EfficientFormer backbone based on .

### Field list

| Field <img width=200/> | Description |
|---|---|
|`name` | (str) Name must be "efficientformer" to use EfficientFormer backbone. |
| `params.num_attention_heads` | (int) |
| `params.attention_hidden_size` | (int) |
| `params.attention_dropout_prob` | (float) |
| `params.attention_ratio` | (int) |
| `params.attention_bias_resolution` | (int) |
| `params.pool_size` | (int) |
| `params.intermediate_ratio` | (int) |
| `params.hidden_dropout_prob` | (float) |
| `params.hidden_activation_type` | (str) |
| `params.layer_norm_eps` | (float) |
| `params.drop_path_rate` | (float) |
| `params.use_layer_scale` | (bool) |
| `params.layer_scale_init_value` | (float) |
| `params.down_patch_size` | (int) |
| `params.down_stride` | (int) |
| `params.down_pad` | (int) |
| `params.vit_num` | (int) |
| `stage_params[n].num_blocks` | (int) |
| `stage_params[n].hidden_sizes` | (int) |
| `stage_params[n].downsamples` | (bool) |