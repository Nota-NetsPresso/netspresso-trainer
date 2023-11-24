import collections
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

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
            bias = nn.functional.interpolate(bias.unsqueeze(0), size=(attention_scores.size(-2), attention_scores.size(-1)), mode='bicubic')
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
