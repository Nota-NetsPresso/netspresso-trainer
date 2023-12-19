"""
Based on the vit implementation of apple/ml-cvnets.
https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/models/classification/vit.py
"""
import argparse
from typing import Any, Dict, Optional, Tuple, Union, List

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
)
from ...op.custom import ConvLayer, SinusoidalPositionalEncoding
from ...utils import BackboneOutput

__all__ = ['vit']
SUPPORTING_TASK = ['classification']

class ViTEmbeddings(nn.Module):
    def __init__(self, image_channels, patch_size, hidden_size, hidden_dropout_prob, use_cls_token=True, vocab_size=1000):
        super().__init__()
        
        image_channels = 3  # {RGB}
        
        kernel_size = patch_size
        if patch_size % 2 == 0:
            kernel_size += 1
        
        self.patch_emb = ConvLayer(
            in_channels=image_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=patch_size,
            bias=True,
            use_norm=False,
            use_act=False,
        )
        self.flat = Image2Sequence(contiguous=True)
        
        self.cls_token = None
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, hidden_size)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        self.pos_embed = SinusoidalPositionalEncoding(
                d_model=hidden_size,
                channels_last=True,
                max_len=vocab_size,
        )
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
    
    def forward(self, x):
        B = x.shape[0]  # B x 3(={RGB}) x H x W

        patch_emb = self.patch_emb(x)  # B x C(=embed_dim) x H'(=patch_size) x W'(=patch_size)
        patch_emb = self.flat(patch_emb)  # B x H'*W' x C

        # add classification token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x C
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)  # B x (H'*W' + 1) x C

        patch_emb = self.pos_embed(patch_emb)  # B x (H'*W' + 1) x C
        
        patch_emb = self.dropout(patch_emb)  # B x (H'*W' + 1) x C
        return patch_emb  # B x (H'*W' + 1) x C

class ViTBlock(MetaFormerBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__(hidden_size, layer_norm_eps)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = MultiHeadAttention(hidden_size, num_attention_heads,
                                                  attention_scale=(hidden_size // num_attention_heads) ** -0.5,
                                                  attention_dropout_prob=attention_dropout_prob,
                                                  use_qkv_bias=True
                                                  )
        self.channel_mlp = ChannelMLP(hidden_size, intermediate_size, hidden_dropout_prob)
    
    
class ViTEncoder(MetaFormerEncoder):
    def __init__(self, num_blocks, hidden_size, num_attention_heads, attention_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ViTBlock(hidden_size, num_attention_heads, attention_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) for _ in range(num_blocks)]
        )

class VisionTransformer(MetaFormer):
    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        patch_size = params.patch_size
        hidden_size = params.attention_channels
        num_blocks = params.num_blocks
        num_attention_heads = params.num_attention_heads
        attention_dropout_prob = params.attention_dropout_prob
        intermediate_size = params.ffn_intermediate_channels
        hidden_dropout_prob = params.ffn_dropout_prob
        use_cls_token = params.use_cls_token
        vocab_size = params.vocab_size

        # Fix as a constant
        layer_norm_eps = 1e-6

        hidden_sizes = hidden_size if isinstance(hidden_size, list) else [hidden_size] * num_blocks
        super().__init__(hidden_sizes)
        self.task = task
        self.intermediate_features = self.task in ['segmentation', 'detection']
        
        image_channels = 3
        self.patch_embed = ViTEmbeddings(image_channels, patch_size, hidden_sizes[-1], hidden_dropout_prob, use_cls_token=use_cls_token, vocab_size=vocab_size)
        self.encoder = ViTEncoder(num_blocks, hidden_sizes[-1], num_attention_heads, attention_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps)
        self.norm = nn.LayerNorm(hidden_sizes[-1], eps=layer_norm_eps)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.patch_embed.cls_token is not None:
            feat = x[:, 0]
        else:
            feat = torch.mean(x, dim=1)
        return BackboneOutput(last_feature=feat)


def vit(task, conf_model_backbone):
    # ViT tiny
    return VisionTransformer(task, conf_model_backbone.params, conf_model_backbone.stage_params)