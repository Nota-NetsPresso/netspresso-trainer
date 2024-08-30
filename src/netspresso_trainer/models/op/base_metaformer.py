# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import collections
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from ..op.custom import LayerScale2d
from ..op.depth import DropPath
from ..op.registry import ACTIVATION_REGISTRY, NORM_REGISTRY
from ..utils import BackboneOutput, FXTensorType


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Image2Sequence(nn.Module):

    def __init__(self, contiguous=False):
        super().__init__()
        self.contiguous = contiguous

    def forward(self, x: Union[Tensor, Proxy]):
        x = x.flatten(2).transpose(1, 2)
        if self.contiguous:
            x = x.contiguous()
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_hidden_size = None,
        value_hidden_size = None,
        attention_scale = None,
        attention_dropout_prob = 0.0,
        use_qkv_bias = True,
        use_attention_bias = False,
        use_cross_attention = False,
        output_with_attentions = False,
        sequence_reduction_ratio = 1,
        attention_bias_resolution = 16,
    ) -> None:
        super().__init__()

        attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        value_hidden_size = value_hidden_size if value_hidden_size is not None else attention_hidden_size

        if attention_hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {attention_hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        if value_hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {value_hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(attention_hidden_size / num_attention_heads)
        self.value_attention_head_size = int(value_hidden_size / num_attention_heads)

        self.head_size = self.num_attention_heads * self.attention_head_size
        self.value_head_size = self.num_attention_heads * self.value_attention_head_size
        self.attention_scale = attention_scale if attention_scale is not None \
            else math.sqrt(self.attention_head_size)


        self.query = nn.Linear(hidden_size, self.head_size, bias=use_qkv_bias)  # ... x C -> ... x C_qk
        self.key = nn.Linear(hidden_size, self.head_size, bias=use_qkv_bias)  # ... x C -> ... x C_qk
        self.value = nn.Linear(hidden_size, self.value_head_size, bias=use_qkv_bias)  # ... x C -> ... x C_v

        self.linear = nn.Linear(self.value_head_size, hidden_size)  # ... x C_v -> ... x C

        self.dropout = nn.Dropout(attention_dropout_prob)
        self.output_with_attentions = output_with_attentions

        self.use_sequence_reduction = False
        if sequence_reduction_ratio > 1:
            self.use_sequence_reduction = True
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.sr_layer_norm = nn.LayerNorm(hidden_size)

        self.use_attention_bias = use_attention_bias
        if self.use_attention_bias:
            # See https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer.py#L48-L61
            assert attention_bias_resolution is not None
            points = list(itertools.product(range(attention_bias_resolution), range(attention_bias_resolution)))
            len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])

            self.attention_biases = torch.nn.Parameter(torch.zeros(self.num_attention_heads, 49))
            self.register_buffer('attention_bias_idxs', torch.ones(49, 49).long())

        #     self.attention_biases_seg = torch.nn.Parameter(
        #         torch.zeros(self.num_attention_heads, len(attention_offsets)))
        #     self.register_buffer('attention_bias_idxs_seg',
        #                          torch.LongTensor(idxs).view(N, N))

        self.use_cross_attention = use_cross_attention

    def transpose_for_scores(self, x: Tensor, attention_head_size: int) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def sequence_reduce(self, x: Tensor, height: int, width: int) -> Tensor:
        """SegFormer
        """
        B, N, C = x.shape
        # Reshape to (batch_size, num_channels, height, width)
        x = x.permute(0, 2, 1).reshape(B, C, height, width)
        # Apply sequence reduction
        x = self.sr(x)
        # Reshape back to (batch_size, seq_len, num_channels)
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.sr_layer_norm(x)
        return x

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        """Forward pass for multi-head attention (self- or cross-)

        Let. S_s(source): S_q(query) (= H'*W' + 1)
        Let. S_t(target): S_k(key) = S_v(value)
        If self-attention, S_s = S_t
        Let. C_qk: C_qksplit * {head}(=num_attention_heads) = (mostly) hidden_size
        Let. C_v: C_vsplit * {head}(=num_attention_heads) * attn_ratio = (mostly) hidden_size * attn_ratio
        Mostly, C = C_qk = C_v (in_channels = hidden_size = value_hidden_size)
        Args:
            query_states (Tensor): [B x S_s x C]
            key_value_states (Optional[Tensor], optional): [B x S_t x C]. Defaults to None.
            head_mask (Optional[Tensor], optional): [B x {head} x S_s x S_t]. Defaults to None.
            height (Optional[int], optional): (Segformer only). Defaults to None.
            width (Optional[int], optional): (Segformer only). Defaults to None.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Tensor]]: output tensor or list of attention matrices
        """

        mixed_query_layer = self.query(query_states)  # B x S_s x C_qk

        if not self.use_cross_attention:  # Self-attention
            key_value_states = query_states  # B x S_t(=S_s) x C_qk
        if self.use_sequence_reduction:
            key_value_states = self.sequence_reduce(key_value_states, height, width)  # B x S_t' x C_qk

        key_layer = self.transpose_for_scores(self.key(key_value_states), self.attention_head_size)  # B x {head} x S_t x C_qksplit
        value_layer = self.transpose_for_scores(self.value(key_value_states), self.value_attention_head_size)  # B x {head} x S_t x C_vsplit
        query_layer = self.transpose_for_scores(mixed_query_layer, self.attention_head_size)  # B x {head} x S_s x C_qksplit

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B x {head} x S_s x S_t

        attention_scores = attention_scores / self.attention_scale  # B x {head} x S_s x S_t

        if self.use_attention_bias:
            bias = self.attention_biases[:, self.attention_bias_idxs]
            bias = nn.functional.interpolate(bias.unsqueeze(0), size=(attention_scores.size(-2), attention_scores.size(-1)), mode='bilinear')
            attention_scores = attention_scores + bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # B x {head} x S_s x S_t

        attention_probs = self.dropout(attention_probs)  # B x {head} x S_s x S_t

        if head_mask is not None:
            attention_probs = attention_probs * head_mask  # B x {head} x S_s x S_t

        context_layer = torch.matmul(attention_probs, value_layer)  # B x {head} x S_s x C_vsplit

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # B x S_s x {head} x C_vsplit
        new_context_layer_shape = context_layer.size()[:-2] + (self.value_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)  # B x S_s x C_v

        context_layer = self.linear(context_layer)  # B x S_s x C
        context_layer = self.dropout(context_layer)  # B x S_s x C

        if self.output_with_attentions:
            return (context_layer, attention_probs)

        return context_layer  # B x S_s x C


class MultiQueryAttention2D(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_hidden_size = None,
        value_hidden_size = None,
        attention_scale = None,
        attention_dropout_prob = 0.0,
        use_qkv_bias = True,
        use_cross_attention = False,
        output_with_attentions = False,
        query_pooling_stride: Optional[int] = None,
        key_val_downsample: Optional[bool] = None,
        key_val_downsample_kernel_size: Optional[int] = None,
        key_val_downsample_stride: Optional[int] = None,
        norm_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        norm_type = norm_type if norm_type is not None else 'batch_norm'
        norm_layer = NORM_REGISTRY[norm_type]

        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        if attention_hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {attention_hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.value_hidden_size = value_hidden_size if value_hidden_size is not None else int(self.attention_hidden_size / num_attention_heads)

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(attention_hidden_size / num_attention_heads)

        self.attention_scale = attention_scale if attention_scale is not None \
            else math.sqrt(self.attention_head_size)

        self.query = []
        if query_pooling_stride:
            self.query.append(nn.AvgPool2d(kernel_size=query_pooling_stride))
            self.query.append(norm_layer(hidden_size))
        self.query.append(nn.Conv2d(hidden_size, self.attention_hidden_size,
                                    kernel_size=1, bias=use_qkv_bias))
        self.query = nn.Sequential(*self.query)

        self.key = []
        self.value = []
        if key_val_downsample:
            self.key.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=key_val_downsample_kernel_size, padding=key_val_downsample_kernel_size // 2,
                                      stride=key_val_downsample_stride, bias=use_qkv_bias, groups=hidden_size))
            self.key.append(norm_layer(hidden_size))
            self.value.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=key_val_downsample_kernel_size, padding=key_val_downsample_kernel_size // 2,
                                        stride=key_val_downsample_stride, bias=use_qkv_bias, groups=hidden_size))
            self.value.append(norm_layer(hidden_size))

        self.key.append(nn.Conv2d(hidden_size, self.attention_head_size, kernel_size=1, bias=use_qkv_bias))
        self.value.append(nn.Conv2d(hidden_size, self.value_hidden_size, kernel_size=1, bias=use_qkv_bias))
        self.key = nn.Sequential(*self.key)
        self.value = nn.Sequential(*self.value)

        self.linear = []
        if query_pooling_stride:
            self.linear.append(nn.Upsample(scale_factor=query_pooling_stride, mode='bilinear', align_corners=False))
        self.linear.append(nn.Conv2d(self.attention_hidden_size, hidden_size, kernel_size=1, bias=False))
        self.linear = nn.Sequential(*self.linear)

        self.dropout = nn.Dropout(attention_dropout_prob)
        self.output_with_attentions = output_with_attentions

        self.use_cross_attention = use_cross_attention

    def transpose_for_scores(self, x: Tensor, num_head: int) -> Tensor:
        x = x.flatten(-2)  # B x C x H x W -> B x C x {H * W}
        new_x_shape = (x.shape[0], num_head,  x.shape[1] // num_head, x.shape[2])
        x = x.view(new_x_shape)  # B x C x {H * W} -> B x {head} x C_split x {H * W}
        return x.permute(0, 1, 3, 2)  # B x {head} x C_split x {H * W} -> B x {head} x {H * W} x C_split

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        B, _, H_q, W_q = query_states.shape  # B x hiddin_dim x H_q x W_q

        query = self.query(query_states)

        if not self.use_cross_attention:  # Self-attention
            key_value_states = query_states

        key = self.key(key_value_states)
        value = self.value(key_value_states)

        # query_states: B x C_q x H_q x W_q
        # key_value_states: B x C_kv x H_kv x W_kv

        key_layer = self.transpose_for_scores(key, 1)  # B x C_kv x H_kv x W_kv -> B x 1 x {H_kv * W_kv} x C_kv
        value_layer = self.transpose_for_scores(value, 1)  # B x C_kv x H_kv x W_kv -> B x 1 x {H_kv * W_kv} x C_kv
        query_layer = self.transpose_for_scores(query, self.num_attention_heads)  # B x C_q x H_q x W_q -> B x {head} x {H_q * W_q} x C_qsplit

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / self.attention_scale

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 1, 3, 2).contiguous()  # B x {head} x {H_q * W*q} x C_qsplit -> B x {head} x C_qsplit x {H_q * W*q}
        context_layer = context_layer.reshape(B, self.attention_hidden_size, H_q, W_q)  # B x {head} x C_qsplit x {H_q * W*q} -> B x C_q x H_q x W*q

        context_layer = self.linear(context_layer)  # B x C_q x H_q x W_q -> B x hiddin_dim x H_q x W_q
        context_layer = self.dropout(context_layer)

        if self.output_with_attentions:
            return (context_layer, attention_probs)

        return context_layer  # B x hiddin_dim x H_q x W_


class MobileMultiQueryAttention2D(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_hidden_size = None,
        value_hidden_size = None,
        attention_scale = None,
        attention_dropout_prob = 0.0,
        use_qkv_bias = True,
        use_cross_attention = False,
        output_with_attentions = False,
        query_pooling_stride: Optional[int] = None,
        key_val_downsample: Optional[bool] = None,
        key_val_downsample_kernel_size: Optional[int] = None,
        key_val_downsample_stride: Optional[int] = None,
        norm_type: Optional[str] = None,
        layer_scale: Optional[float] = None,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        norm_type = norm_type if norm_type is not None else 'batch_norm'
        self.norm_layer = NORM_REGISTRY[norm_type](hidden_size)
        self.attention = MultiQueryAttention2D(
            hidden_size, num_attention_heads, attention_hidden_size, value_hidden_size, attention_scale, attention_dropout_prob,
            use_qkv_bias, use_cross_attention, output_with_attentions, query_pooling_stride, key_val_downsample,
            key_val_downsample_kernel_size, key_val_downsample_stride, norm_type
        )
        self.apply_layer_scale = False
        if layer_scale is not None:
            self.apply_layer_scale = True
            self.layer_scale = LayerScale2d(hidden_size, layer_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        residual = query_states

        out = self.norm_layer(query_states)
        out = self.attention(out, key_value_states, head_mask)
        if self.apply_layer_scale:
            out = self.layer_scale(out)
        out = self.drop_path(out) + residual
        return out


class ChannelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob, hidden_activation_type='silu'):
        super().__init__()
        self.ffn = nn.Sequential()
        self.ffn.add_module('dense1', nn.Linear(in_features=hidden_size, out_features=intermediate_size, bias=True))
        self.ffn.add_module('act', ACTIVATION_REGISTRY[hidden_activation_type]())
        self.ffn.add_module('dropout', nn.Dropout(p=hidden_dropout_prob))
        self.ffn.add_module('dense2', nn.Linear(in_features=intermediate_size, out_features=hidden_size, bias=True))

        self.dropout = nn.Dropout(p=hidden_dropout_prob)

    def forward(self, x):
        x = self.ffn(x)
        x = self.dropout(x)
        return x

class MetaFormerBlock(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.layernorm_after = nn.LayerNorm(hidden_size)
        self.token_mixer = nn.Identity()  # MultiHeadAttention()
        self.channel_mlp = nn.Identity()  # ChannelMLP()

    def forward(self, x):
        out_token_mixer = self.layernorm_before(x)
        out_token_mixer = self.token_mixer(out_token_mixer)

        out_token_mixer = out_token_mixer + x

        out_final = self.layernorm_after(out_token_mixer)
        out_final = self.channel_mlp(out_final)

        out_final = out_final + out_token_mixer

        return out_final

class MetaFormerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Identity()
        # self.blocks = nn.Sequential(
        #     *[MetaFormerBlock(hidden_size, layer_norm_eps) for _ in range(num_layers)]
        # )

    def forward(self, x):
        x = self.blocks(x)
        return x

class MetaFormer(nn.Module):
    def __init__(self, hidden_sizes) -> None:
        super().__init__()
        self._feature_dim = hidden_sizes[-1]
        self._intermediate_features_dim = hidden_sizes

        self.patch_embed = nn.Identity()
        self.encoder = MetaFormerEncoder()
        self.norm = nn.Identity()

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def forward(self, x: FXTensorType):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        feat = torch.mean(x, dim=1)
        return BackboneOutput(last_feature=feat)
