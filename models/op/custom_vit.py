from typing import Optional, Tuple, List, Any, Union, Dict
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from models.op.swish import Swish


NORM_REGISTRY: Dict[str, nn.Module] = {
    'batch_norm': nn.BatchNorm2d,
    'instance_norm': nn.InstanceNorm2d,
}

ACTIVATION_REGISTRY: Dict[str, nn.Module] = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'leaky_relu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'swish': Swish
}

class ViTSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob = 0.0,
        use_qkv_bias = True,
        embedding_size = None,
        output_attentions = False
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0 and embedding_size is None:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=use_qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=use_qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=use_qkv_bias)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_attentions = output_attentions

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.output_attentions:
            return (context_layer, attention_probs)
        
        # return (context_layer,)
        return context_layer