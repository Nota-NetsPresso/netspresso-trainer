"""
EfficientFormer implementation
"""
import copy
import itertools
import math
import os
from typing import Dict, Optional, List

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ...op.base_metaformer import (
    ChannelMLP,
    Image2Sequence,
    MetaFormer,
    MetaFormerBlock,
    MetaFormerEncoder,
    MultiHeadAttention,
    Pooling,
)
from ...op.custom import ConvLayer
from ...op.depth import DropPath
from ...utils import BackboneOutput

SUPPORTING_TASK = ['classification', 'segmentation', 'detection']


class EfficientFormerStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            ConvLayer(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=True,
                      use_act=True, use_norm=True, norm_type='batch_norm', act_type='relu'),
            ConvLayer(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=True,
                      use_act=True, use_norm=True, norm_type='batch_norm', act_type='relu'),
        )

    def forward(self, x):
        x = self.stem(x)
        return x


class EfficientFormerEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.proj = ConvLayer(
            in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True,
            use_norm=True, use_act=False, norm_type='batch_norm'
        )

    def forward(self, x):
        x = self.proj(x)  # B x C x H//stride x W//stride
        return x


class EfficientFormerMeta4DMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None, hidden_dropout_prob=0., hidden_activation_type='gelu'):
        # hidden_size: in_features
        # intermediate_size: hidden_features
        # hidden_activation_type = 'gelu'
        # hidden_dropout_prob = drop = 0.
        super().__init__()
        intermediate_size = intermediate_size or hidden_size

        self.ffn = nn.Sequential()
        self.ffn.add_module(
            'conv_1x1_1', ConvLayer(hidden_size, intermediate_size, kernel_size=1, bias=True,
                                    use_norm=True, use_act=True, norm_type='batch_norm',
                                    act_type=hidden_activation_type)
        )
        self.ffn.add_module('drop', nn.Dropout(p=hidden_dropout_prob))
        self.ffn.add_module(
            'conv_1x1_2', ConvLayer(intermediate_size, hidden_size, kernel_size=1, bias=True,
                                    use_norm=True, use_act=False, norm_type='batch_norm')
        )

        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ffn(x)
        x = self.dropout(x)
        return x


class Meta3D(MetaFormerBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_hidden_size,
                 attention_dropout_prob, attention_ratio, attention_bias_resolution,
                 intermediate_ratio, hidden_dropout_prob, hidden_activation_type='gelu',
                 layer_norm_eps=1e-5,
                 drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__(hidden_size, layer_norm_eps)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # intermediate_ratio: mlp_ratio
        # hidden_activation_type: act_type
        # hidden_dropout_prob: drop
        # hidden_size: dim
        # hidden_size: key_dim * num_attention_heads

        # self.token_mixer = Attention(dim)
        # (dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=16)
        # num_attention_heads: 8
        # hidden_size: key_dim*num_attention_heads = 32 * 8
        # value_hidden_size: key_dim*num_attention_heads*attn_ratio = 32 * 8 * 4
        # attention_bias_resolution: resolution = 16
        self.token_mixer = MultiHeadAttention(
            hidden_size, num_attention_heads,
            attention_hidden_size=attention_hidden_size,
            attention_dropout_prob=attention_dropout_prob,
            value_hidden_size=attention_hidden_size*attention_ratio,
            use_attention_bias=True,
            attention_bias_resolution=attention_bias_resolution
        )
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        intermediate_size = int(hidden_size * intermediate_ratio)
        self.channel_mlp = ChannelMLP(
            hidden_size, intermediate_size, hidden_dropout_prob, hidden_activation_type
        )

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_size)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_size)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) *
                self.token_mixer(self.layernorm_before(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) *
                self.channel_mlp(self.layernorm_after(x))
            )

        else:
            x = x + self.drop_path(self.token_mixer(self.layernorm_before(x)))
            x = x + self.drop_path(self.channel_mlp(self.layernorm_after(x)))
        return x


class Meta4D(MetaFormerBlock):
    def __init__(self, hidden_size, pool_size=3, intermediate_ratio=4.,
                 hidden_dropout_prob=0., hidden_activation_type='gelu',
                 drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        # intermediate_ratio: mlp_ratio
        # hidden_size: dim
        # hidden_dropout_prob: drop
        # hidden_activation_type: act_type
        super().__init__(hidden_size, 0.)
        self.layernorm_before = nn.Identity()  # not used
        self.layernorm_after = nn.Identity()  # not used
        self.token_mixer = Pooling(pool_size)
        intermediate_size = int(hidden_size * intermediate_ratio)
        self.channel_mlp = EfficientFormerMeta4DMLP(
            hidden_size, intermediate_size, hidden_dropout_prob, hidden_activation_type
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_size)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_size)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(x)
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.channel_mlp(x)
            )
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.channel_mlp(x))
        return x


class EfficientFormerEncoder(MetaFormerEncoder):
    def __init__(self, use_intermediate_features, num_blocks, hidden_sizes,
                 num_attention_heads, attention_hidden_size, attention_dropout_prob,
                 attention_ratio, attention_bias_resolution,
                 pool_size, intermediate_ratio, hidden_dropout_prob, hidden_activation_type,
                 layer_norm_eps,
                 drop_path_rate, use_layer_scale, layer_scale_init_value,
                 downsamples, down_patch_size, down_stride, down_pad,
                 vit_num=1):
        # intermediate_ratio: mlp_ratio
        # hidden_size: dim
        # hidden_dropout_prob: drop
        # hidden_activation_type: act_type

        # intermediate_ratio: mlp_ratio
        # hidden_activation_type: act_type
        # hidden_dropout_prob: drop
        # hidden_size: dim
        # attention_hidden_size: key_dim * num_attention_heads

        # self.token_mixer = Attention(dim)
        # (dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=16)
        # num_attention_heads: 8
        # hidden_size: key_dim*num_attention_heads = 32 * 8
        # value_hidden_size: key_dim*num_attention_heads*attn_ratio = 32 * 8 * 4
        # attention_bias_resolution: resolution = 16

        super().__init__()
        self.use_intermediate_features = use_intermediate_features

        blocks = []
        for module_idx in range(len(num_blocks)):
            stage = self.meta_blocks(
                num_blocks, module_idx, hidden_sizes[module_idx],
                num_attention_heads, attention_hidden_size, attention_dropout_prob,
                attention_ratio, attention_bias_resolution,
                pool_size, intermediate_ratio, hidden_dropout_prob, hidden_activation_type,
                layer_norm_eps,
                drop_path_rate, use_layer_scale, layer_scale_init_value,
                vit_num
            )
            blocks.append(stage)
            if module_idx >= len(num_blocks) - 1:
                break
            if downsamples[module_idx] or hidden_sizes[module_idx] != hidden_sizes[module_idx + 1]:
                # downsampling between two stages
                blocks.append(
                    EfficientFormerEmbedding(
                        hidden_sizes[module_idx],
                        hidden_sizes[module_idx + 1],
                        down_patch_size,
                        down_stride,
                        down_pad
                    )
                )

        self.blocks = nn.ModuleList(blocks)

        if self.use_intermediate_features:
            # add a norm layer for each output
            self.intermediate_features_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.intermediate_features_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = nn.GroupNorm(1, hidden_sizes[i_emb])
                self.add_module(f'norm{i_layer}', layer)

    @staticmethod
    def meta_blocks(num_blocks, module_idx, hidden_size,
                    num_attention_heads, attention_hidden_size, attention_dropout_prob,
                    attention_ratio, attention_bias_resolution,
                    pool_size, intermediate_ratio, hidden_dropout_prob, hidden_activation_type,
                    layer_norm_eps,
                    drop_path_rate, use_layer_scale, layer_scale_init_value,
                    vit_num=1):

        blocks = []
        if module_idx == 3 and vit_num == num_blocks[module_idx]:
            blocks.append(Image2Sequence())
        for block_idx in range(num_blocks[module_idx]):
            block_dpr = drop_path_rate * (block_idx + sum(num_blocks[:module_idx])) / (sum(num_blocks) - 1)
            if module_idx == 3 and num_blocks[module_idx] - block_idx <= vit_num:
                blocks.append(Meta3D(
                    hidden_size, num_attention_heads, attention_hidden_size,
                    attention_dropout_prob,
                    attention_ratio,
                    attention_bias_resolution,
                    intermediate_ratio,
                    hidden_dropout_prob,
                    hidden_activation_type,
                    layer_norm_eps,
                    block_dpr,
                    use_layer_scale,
                    layer_scale_init_value
                ))
            else:
                blocks.append(Meta4D(
                    hidden_size, pool_size, intermediate_ratio,
                    hidden_dropout_prob,
                    hidden_activation_type,
                    block_dpr,
                    use_layer_scale,
                    layer_scale_init_value
                ))
                if module_idx == 3 and num_blocks[module_idx] - block_idx - 1 == vit_num:
                    blocks.append(Image2Sequence())

        blocks = nn.Sequential(*blocks)
        return blocks

    def forward(self, x):
        all_hidden_states = () if self.use_intermediate_features else None
        for idx, block in enumerate(self.blocks):
            # if len(x.size()) == 4:
            B, C, H, W = x.size()
            x = block(x)
            if self.use_intermediate_features and idx in self.intermediate_features_indices:
                norm_layer = getattr(self, f'norm{idx}')
                # @deepkyu: [fx tracing] len(x.size()) == 3 when idx == 6 (with Meta3D, in efficientformer_l1)
                # if len(x.size()) != 4:
                if idx == self.intermediate_features_indices[-1]:
                    x = x.transpose(1, 2).reshape(B, C, H, W)
                x = norm_layer(x)
                all_hidden_states = all_hidden_states + (x,)

        if self.use_intermediate_features:
            # output the features of four stages for dense prediction
            return all_hidden_states
        # output only the features of last layer for image classification
        return x


class EfficientFormer(MetaFormer):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        
        num_blocks = [stage.num_blocks for stage in stage_params]
        hidden_sizes = [stage.channels for stage in stage_params]

        num_attention_heads = params.num_attention_heads
        attention_hidden_size = params.attention_channels
        attention_dropout_prob = params.attention_dropout_prob
        attention_ratio = params.attention_value_expansion_ratio
        intermediate_ratio = params.ffn_intermediate_ratio
        hidden_dropout_prob = params.ffn_dropout_prob
        hidden_activation_type = params.ffn_act_type
        vit_num = params.vit_num

        # Fix as constant
        layer_norm_eps = 1e-5
        layer_scale_init_value = 1e-5
        down_patch_size = 3
        down_stride = 2
        down_pad = 1
        pool_size = 3
        downsamples = [True for _ in stage_params]
        use_layer_scale = True
        attention_bias_resolution = 16
        drop_path_rate = 0.

        super().__init__(hidden_sizes)
        self.task = task.lower()
        self.use_intermediate_features = self.task in ['segmentation', 'detection']

        image_channels = 3
        self.patch_embed = EfficientFormerStem(in_channels=image_channels, out_channels=hidden_sizes[0])

        self.encoder = EfficientFormerEncoder(
            self.use_intermediate_features, num_blocks, hidden_sizes,
            num_attention_heads, attention_hidden_size, attention_dropout_prob,
            attention_ratio, attention_bias_resolution,
            pool_size, intermediate_ratio, hidden_dropout_prob, hidden_activation_type,
            layer_norm_eps,
            drop_path_rate, use_layer_scale, layer_scale_init_value,
            downsamples, down_patch_size, down_stride, down_pad,
            vit_num=vit_num
        )

        self.norm = nn.LayerNorm(hidden_sizes[-1], eps=layer_norm_eps)

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        if self.use_intermediate_features:
            all_hidden_states = x  # (features)
            return BackboneOutput(intermediate_features=all_hidden_states)
        x = self.norm(x)  # B x N x C
        feat = torch.mean(x, dim=-2)
        return BackboneOutput(last_feature=feat)


def efficientformer(task, conf_model_backbone) -> EfficientFormer:
    return EfficientFormer(task, conf_model_backbone.params, conf_model_backbone.stage_params)
