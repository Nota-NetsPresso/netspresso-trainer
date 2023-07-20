from typing import Optional, Tuple, List, Any, Union, Dict
import math
import itertools
import collections

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_scale = None,
        attention_dropout_prob = 0.0,
        use_qkv_bias = True,
        use_attention_bias = False,
        use_cross_attention = False,
        output_with_attentions = False
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_scale = attention_scale if attention_scale is not None \
            else math.sqrt(self.attention_head_size)


        self.query = nn.Linear(hidden_size, self.all_head_size, bias=use_qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=use_qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=use_qkv_bias)
        
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_dropout_prob)
        self.output_with_attentions = output_with_attentions


        self.use_attention_bias = use_attention_bias
        # TODO: add attention bias
        # if self.use_attention_bias:
        #     # See https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer.py#L48-L61
        #     resolution = 16
        #     points = list(itertools.product(range(resolution), range(resolution)))
        #     N = len(points)
        #     attention_offsets = {}
        #     idxs = []
        #     for p1 in points:
        #         for p2 in points:
        #             offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        #             if offset not in attention_offsets:
        #                 attention_offsets[offset] = len(attention_offsets)
        #             idxs.append(attention_offsets[offset])

        #     self.register_buffer('attention_biases',
        #                          torch.zeros(self.num_attention_heads, 49))
        #     self.register_buffer('attention_bias_idxs',
        #                          torch.ones(49, 49).long())

        #     self.attention_biases_seg = torch.nn.Parameter(
        #         torch.zeros(self.num_attention_heads, len(attention_offsets)))
        #     self.register_buffer('attention_bias_idxs_seg',
        #                          torch.LongTensor(idxs).view(N, N))
    
        self.use_cross_attention = use_cross_attention

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        # Let S_s(source): S_q(query) (= H'*W' + 1)
        # Let S_t(target): S_k(key) = S_v(value)
        # If self-attention, S_s = S_t
        # Let C: C_split * {head}(=num_attention_heads) = (mostly) hidden_size
        # query_states: B x S_s x C
        mixed_query_layer = self.query(query_states)  # B x S_s x C
        
        if not self.use_cross_attention:  # Self-attention
            key_value_states = query_states  # B x S_t(=S_s) x C

        key_layer = self.transpose_for_scores(self.key(key_value_states))  # B x {head} x S_t x C_split
        value_layer = self.transpose_for_scores(self.value(key_value_states))  # B x {head} x S_t x C_split
        query_layer = self.transpose_for_scores(mixed_query_layer)  # B x {head} x S_s x C_split

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B x {head} x S_s x S_t

        attention_scores = attention_scores / self.attention_scale  # B x {head} x S_s x S_t

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # B x {head} x S_s x S_t

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # B x {head} x S_s x S_t

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask  # B x {head} x S_s x S_t

        context_layer = torch.matmul(attention_probs, value_layer)  # B x {head} x S_s x C_split

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # B x S_s x {head} x C_split
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)  # B x S_s x C
        
        context_layer = self.linear(context_layer)  # B x S_s x C
        context_layer = self.dropout(context_layer)  # B x S_s x C

        if self.output_with_attentions:
            return (context_layer, attention_probs)
        
        return context_layer  # B x S_s x C
    
class ChannelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super().__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(in_features=hidden_size, out_features=intermediate_size, bias=True),
            nn.SiLU(inplace=False),
            nn.Linear(in_features=intermediate_size, out_features=hidden_size, bias=True),
        ])
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
    
    def forward(self, x):
        x = self.ffn(x)
        x = self.dropout(x)
        return x

class MetaFormerBlock(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
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
    def __init__(self, num_layers, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self._last_channels = hidden_size
        
        self.patch_embed = nn.Identity()
        self.encoder = MetaFormerEncoder()
        self.norm = nn.Identity()

    @property
    def last_channels(self):
        return self._last_channels
        
    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x
    
    def forward_tokens(self, x):
        x = self.encoder(x)
        return x
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        feat = torch.mean(x, dim=1)
        return {'last_feature': feat}