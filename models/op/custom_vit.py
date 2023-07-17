from typing import Optional, Tuple, List, Any, Union, Dict
import math
import itertools
import collections

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from models.op.base_metaformer import MetaFormer, MetaFormerBlock
from models.registry import NORM_REGISTRY, ACTIVATION_REGISTRY


class ViTPatchEmbeddings(nn.Module):
    """
    Refer to https://github.com/huggingface/transformers/blob/9a5d468ba0562e2d5edf9da787881fa227132bca/src/transformers/models/vit/modeling_vit.py#L146C1-L182C26
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, image_size, patch_size, in_channels, hidden_size):
        super().__init__()

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, C, H, W = pixel_values.shape

        # (H, W) should be matched with self.image_size
        # assert H == self.image_size[0] and W == self.image_size[1]

        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

class ViTEmbeddings(nn.Module):
    """
    Refer to https://github.com/huggingface/transformers/blob/9a5d468ba0562e2d5edf9da787881fa227132bca/src/transformers/models/vit/modeling_vit.py#L66C1-L143C26
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_scale = None,
        attention_probs_dropout_prob = 0.0,
        use_qkv_bias = True,
        use_attention_bias = False,
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

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
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
    

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        value_states: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        mixed_query_layer = self.query(query_states)
        
        if key_value_states is None:  # Self-attention
            key_value_states = query_states

        key_layer = self.transpose_for_scores(self.key(key_value_states))
        value_layer = self.transpose_for_scores(self.value(key_value_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / self.attention_scale

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

        if self.output_with_attentions:
            return (context_layer, attention_probs)
        
        # return (context_layer,)
        return context_layer
    
class ViTBlock(MetaFormerBlock):
    def __init__(self, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = nn.Identity()  # TODO: define token mixer
        self.channel_mlp = nn.Identity()  # TODO: define channel nlp
    
    # def forward(self, x):
    #     out_token_mixer = self.layernorm_before(x)
    #     out_token_mixer = self.token_mixer(out_token_mixer)
        
    #     out_token_mixer = out_token_mixer + x
        
    #     out_final = self.layernorm_after(out_token_mixer)
    #     out_final = self.channel_mlp(out_final)
        
    #     out_final = out_final + out_token_mixer
        
    #     return out_final
        

class ViT(MetaFormer):
    def __init__(self, num_layers, hidden_size, layer_norm_eps) -> None:
        super().__init__()
        self.patch_embed = nn.Identity()
        self.blocks = nn.ModuleList(
            [MetaFormerBlock(hidden_size, layer_norm_eps) for _ in range(num_layers)]
        )
        self.norm = nn.Identity()
        
    # def forward_embeddings(self, x):
    #     x = self.patch_embed(x)
    #     return x
    
    # def forward_tokens(self, x):
    #     for block_idx, block in enumerate(self.blocks):
    #         x = block(x)
    #     return x
    
    # def forward(self, x):
    #     x = self.forward_embeddings(x)
    #     x = self.forward_tokens(x)
    #     x = self.norm(x)
    #     return x