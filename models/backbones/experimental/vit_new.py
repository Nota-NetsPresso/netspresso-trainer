"""
Based on the vit implementation of apple/ml-cvnets.
https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/models/classification/vit.py
"""
import argparse
from typing import Union, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

from models.configuration.vit import get_configuration
from models.op.ml_cvnets import ConvLayer, LinearLayer, SinusoidalPositionalEncoding
from models.op.ml_cvnets import TransformerEncoder
from models.op.base_metaformer import MetaFormer, MetaFormerBlock, MetaFormerEncoder

__all__ = ['vit']
SUPPORTING_TASK = ['classification']



class ViTEmbeddings(nn.Module):
    def __init__(self, opts, image_channels, patch_size, hidden_size, hidden_dropout_prob, use_cls_token=True, vocab_size=1000):
        
        image_channels = 3  # {RGB}
        
        vit_config = get_configuration()
        patch_size = vit_config["patch_size"]
        hidden_size = vit_config["embed_dim"]
        # ffn_dim = vit_config["ffn_dim"]
        hidden_dropout_prob = vit_config["pos_emb_drop_p"]
        # n_transformer_layers = vit_config["n_transformer_layers"]
        # num_heads = vit_config["n_attn_heads"]
        # attn_dropout = vit_config["attn_dropout"]
        # dropout = vit_config["dropout"]
        # ffn_dropout = vit_config["ffn_dropout"]
        # norm_layer = vit_config["norm_layer"]
        
        kernel_size = patch_size
        if patch_size % 2 == 0:
            kernel_size += 1
        
        self.patch_emb = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=patch_size,
            bias=True,
            use_norm=False,
            use_act=False,
        )
        
        # use_cls_token = not getattr(
        #     opts, "model.classification.vit.no_cls_token", False
        # )
        
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
    
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]  # B x 3(={RGB}) x H x W

        patch_emb = self.patch_emb(x)  # B x C(=embed_dim) x H'(=patch_size) x W'(=patch_size)
        patch_emb = patch_emb.flatten(2)  # B x C x H'*W'
        patch_emb = patch_emb.transpose(1, 2).contiguous()  # B x H'*W' x C

        # add classification token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x C
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)  # B x (H'*W' + 1) x C

        patch_emb = self.pos_embed(patch_emb)  # B x (H'*W' + 1) x C
        
        patch_emb = self.dropout(patch_emb)  # B x (H'*W' + 1) x C
        return patch_emb  # B x (H'*W' + 1) x C

class ViTChannelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        self.pre_norm_ffn = nn.ModuleList([
            LinearLayer(in_features=hidden_size, out_features=intermediate_size, bias=True),
            nn.SiLU(inplace=False),
            LinearLayer(in_features=intermediate_size, out_features=hidden_size, bias=True),
            nn.Dropout(p=hidden_dropout_prob),
        ])
    
    def forward(self, x):
        for layer in self.pre_norm_ffn:
            x = layer(x)
        return x

class ViTBlock(MetaFormerBlock):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = nn.Identity()  # TODO: define token mixer
        self.channel_mlp = ViTChannelMLP(hidden_size, intermediate_size, hidden_dropout_prob)
    
    # def forward(self, x):
    #     out_token_mixer = self.layernorm_before(x)
    #     out_token_mixer = self.token_mixer(out_token_mixer)
        
    #     out_token_mixer = out_token_mixer + x
        
    #     out_final = self.layernorm_after(out_token_mixer)
    #     out_final = self.channel_mlp(out_final)
        
    #     out_final = out_final + out_token_mixer
        
    #     return out_final
    
    
class ViTEncoder(MetaFormerEncoder):
    def __init__(self, num_layers, hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps) for _ in range(num_layers)]
        )
    
    # def forward(self, x):
    #     for block_idx, block in enumerate(self.blocks):
    #         x = block(x)
    #     return x

class VisionTransformer(MetaFormer):
    def __init__(
        self,
        task,
        opts,
        image_channels,
        patch_size,
        hidden_size,
        num_hidden_layers,
        intermediate_size,
        hidden_dropout_prob,
        layer_norm_eps=1e-6,
        use_cls_token=True,
        vocab_size=1000
    ) -> None:
        super().__init__()
        self.patch_embed = ViTEmbeddings(opts, image_channels, patch_size, hidden_size, hidden_dropout_prob, use_cls_token=use_cls_token, vocab_size=vocab_size)
        self.encoder = ViTEncoder(num_hidden_layers, hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

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
        return x





def vit(task, *args, **kwargs):
    return VisionTransformer(task, opts=None)