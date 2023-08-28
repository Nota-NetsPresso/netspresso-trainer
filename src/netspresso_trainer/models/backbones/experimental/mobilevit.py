"""
Based on the mobilevit official implementation.
https://github.com/apple/ml-cvnets/blob/6acab5e446357cc25842a90e0a109d5aeeda002f/cvnets/models/classification/mobilevit.py
"""

import argparse
from typing import Dict, Tuple, Optional, Any, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...op.ml_cvnets import ConvLayer, GlobalPool
from ...op.ml_cvnets import InvertedResidual
from ...op.base_metaformer import MetaFormer, MetaFormerBlock, MetaFormerEncoder, MultiHeadAttention, ChannelMLP
from ...utils import FXTensorType, BackboneOutput

__all__ = ['mobilevit']
SUPPORTING_TASK = ['classification']

class MobileViTEmbeddings(nn.Module):
    def __init__(self, image_channels, hidden_size):
        super().__init__()
        
        image_channels = 3  # {RGB}
        self.conv = ConvLayer(
            opts=None,
            in_channels=image_channels,
            out_channels=hidden_size,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
    
    def forward(self, x):
        # x: B x 3(={RGB}) x H x W
        x = self.conv(x)  # B x C x H//2 x W//2
        return x

class MobileViTTransformerBlock(MetaFormerBlock):
    # Original: TransformerEncoder
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__(hidden_size, layer_norm_eps)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = MultiHeadAttention(hidden_size, num_attention_heads,
                                              attention_scale=(hidden_size // num_attention_heads) ** -0.5,
                                              attention_dropout_prob=attention_dropout_prob,
                                              use_qkv_bias=True)
        self.channel_mlp = ChannelMLP(hidden_size, intermediate_size, hidden_dropout_prob)

class MobileViTBlock(nn.Module):
    # Original: MobileViTBlock
    def __init__(self, num_transformer_blocks, in_channels, hidden_size, local_kernel_size,
                 patch_h, patch_w, num_attention_heads, attention_dropout_prob, intermediate_size, hidden_dropout_prob,
                 layer_norm_eps, use_fusion_layer=True) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_h * self.patch_w
        
        self.local_rep = nn.Sequential(*[
            ConvLayer(opts=None, in_channels=in_channels, out_channels=in_channels,
                      kernel_size=local_kernel_size, stride=1, 
                      use_norm=True, use_act=True),
            ConvLayer(opts=None, in_channels=in_channels, out_channels=hidden_size,
                      kernel_size=1, stride=1,
                      use_norm=False, use_act=False)
        ])
        
        global_rep_blocks = [
            MobileViTTransformerBlock(
                hidden_size=hidden_size,  # embed_dim, transformer_dim
                num_attention_heads=num_attention_heads,  # num_heads
                attention_dropout_prob=attention_dropout_prob,  # attn_dropout
                intermediate_size=intermediate_size,  # ffn_latent_dim, ffn_dim
                hidden_dropout_prob=hidden_dropout_prob,  # ffn_dropout
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_transformer_blocks)
        ]
        global_rep_blocks.append(nn.LayerNorm(normalized_shape=hidden_size))
        self.global_rep = nn.Sequential(*global_rep_blocks)
        
        self.proj = ConvLayer(opts=None, in_channels=hidden_size, out_channels=in_channels,
                              kernel_size=1, stride=1,
                              use_norm=True, use_act=True)
        
        self.use_fusion_layer = use_fusion_layer
        if self.use_fusion_layer:
            self.fusion = ConvLayer(opts=None, in_channels=in_channels * 2, out_channels=in_channels,
                                    kernel_size=local_kernel_size, stride=1,
                                    use_norm=True, use_act=True)
        
    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        batch_size, in_channels, orig_h, orig_w = feature_map.size()

        # @deepkyu: [fx tracing] remove ceil becuase mod == 0, while ceil is not supported mostly in fx
        # new_h = math.ceil(orig_h / self.patch_h) * self.patch_h
        # new_w = math.ceil(orig_w / self.patch_w) * self.patch_w
        
        new_h = orig_h // self.patch_h * self.patch_h
        new_w = orig_w // self.patch_w * self.patch_w

        interpolate = False
        # @deepkyu: [fx tracing] Found always satisfied: (new_w == orig_w) and (new_h == orig_h)
        # if new_w != orig_w or new_h != orig_h:
        #     # Note: Padding can be done, but then it needs to be handled in attention function.
        #     feature_map = F.interpolate(
        #         feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
        #     )
        #     interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // self.patch_w  # n_w
        num_patch_h = new_h // self.patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B, C, n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size, in_channels, num_patch_h, self.patch_h, num_patch_w, self.patch_w
        )
        # [B, C, n_h, p_h, n_w, p_w] --> [B, C, n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(3, 4)
        # [B, C, n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, self.patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * self.patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        
        # @deepkyu: [fx tracing] Found always satisfied: assert n_dim == 3
        # assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
        #     patches.shape
        # )
        
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B, C, n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size, channels, num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        # [B, C, n_h, n_w, p_h, p_w] --> [B, C, n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(3, 4)
        # [B, C, n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    # def forward_spatial(self, x: Tensor) -> Tensor:
    #     res = x

    #     fm = self.local_rep(x)

    #     # convert feature map to patches
    #     patches, info_dict = self.unfolding(fm)

    #     # learn global representations
    #     for transformer_layer in self.global_rep:
    #         patches = transformer_layer(patches)

    #     # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
    #     fm = self.folding(patches=patches, info_dict=info_dict)

    #     fm = self.conv_proj(fm)

    #     if self.fusion is not None:
    #         fm = self.fusion(torch.cat((res, fm), dim=1))
    #     return fm

    # def forward_temporal(
    #     self, x: Tensor, x_prev: Optional[Tensor] = None
    # ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    #     res = x
    #     fm = self.local_rep(x)

    #     # convert feature map to patches
    #     patches, info_dict = self.unfolding(fm)

    #     # learn global representations
    #     for global_layer in self.global_rep:
    #         if isinstance(global_layer, TransformerEncoder):
    #             patches = global_layer(x=patches, x_prev=x_prev)
    #         else:
    #             patches = global_layer(patches)

    #     # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
    #     fm = self.folding(patches=patches, info_dict=info_dict)

    #     fm = self.conv_proj(fm)

    #     if self.fusion is not None:
    #         fm = self.fusion(torch.cat((res, fm), dim=1))
    #     return fm, patches

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        out = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(out)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        out = self.folding(patches=patches, info_dict=info_dict)

        out = self.proj(out)

        if self.use_fusion_layer:
            out = self.fusion(torch.cat((x, out), dim=1))  # skip concat
        return out

class MobileViTEncoder(MetaFormerEncoder):
    def __init__(self, config_stages, patch_embedding_out_channels, local_kernel_size, patch_size, num_attention_heads, attention_dropout_prob, hidden_dropout_prob, layer_norm_eps, use_fusion_layer) -> None:
        super().__init__()
        stages = []
        
        self.dilation = 1
        self.local_kernel_size = local_kernel_size
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.use_fusion_layer = use_fusion_layer
        
        in_channels = patch_embedding_out_channels
        for config_stage in config_stages:
            stages.append(self._make_block(config_stage, in_channels))
            in_channels = config_stage['out_channels']
        self.blocks = nn.Sequential(*stages)
    
    def _make_block(self, config_stage, in_channels):
        block_type = config_stage['block_type']
        
        out_channels = config_stage['out_channels']
        stride = config_stage['stride']
        expand_ratio = config_stage['expand_ratio']
        if block_type.lower() == 'mobilevit':
            dilate = config_stage['dilate']
            num_transformer_blocks = config_stage['num_transformer_blocks']
            hidden_size = config_stage['hidden_size']
            intermediate_size = config_stage['intermediate_size']
            return self._make_mobilevit_blocks(
                num_transformer_blocks, in_channels, out_channels, stride, expand_ratio,
                hidden_size, intermediate_size, self.patch_size, self.local_kernel_size, self.num_attention_heads,
                self.attention_dropout_prob, self.hidden_dropout_prob, self.layer_norm_eps, self.use_fusion_layer,
                dilate
            )
            
        num_blocks = config_stage['num_blocks']
        return self._make_inverted_residual_blocks(
            num_blocks, in_channels, out_channels, stride, expand_ratio
        )
    
    def _make_inverted_residual_blocks(self, num_blocks, in_channels, out_channels, stride, expand_ratio):
        blocks = [
            InvertedResidual(
                opts=None,
                in_channels=in_channels if block_idx == 0 else out_channels,
                out_channels=out_channels,
                stride=stride if block_idx == 0 else 1,
                expand_ratio=expand_ratio,
                dilation=1
            ) for block_idx in range(num_blocks)
        ]
        return nn.Sequential(*blocks)
    
    def _make_mobilevit_blocks(self, num_transformer_blocks, in_channels, out_channels, stride, expand_ratio,
                               hidden_size, intermediate_size, patch_size, local_kernel_size, num_attention_heads, attention_dropout_prob,
                               hidden_dropout_prob, layer_norm_eps, use_fusion_layer, dilate: Optional[bool] = False):
        blocks = []
        if stride == 2:
            blocks.append(
                InvertedResidual(
                    opts=None,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride if not dilate else 1,
                    expand_ratio=expand_ratio,
                    dilation=self.dilation
                )
            )
            if dilate:
                self.dilation += 2
                
            in_channels = out_channels
            
        blocks.append(
            MobileViTBlock(
                num_transformer_blocks=num_transformer_blocks,
                in_channels=in_channels,
                hidden_size=hidden_size,
                local_kernel_size=local_kernel_size,  # getattr(opts, "model.classification.mit.conv_kernel_size", 3)
                patch_h=patch_size,  # patch_h
                patch_w=patch_size,  # patch_w
                num_attention_heads=num_attention_heads,  # num_heads
                attention_dropout_prob=attention_dropout_prob,  # attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.1)
                intermediate_size=intermediate_size,  # ffn_dim
                hidden_dropout_prob=hidden_dropout_prob,  # dropout, ffn_dropout
                layer_norm_eps=layer_norm_eps,
                use_fusion_layer=use_fusion_layer  # not getattr(opts, "model.classification.mit.no_fuse_local_global_features", False)
            )
        )
        
        return nn.Sequential(*blocks)
    

class MobileViT(MetaFormer):
    def __init__(
        self,
        task,
        config_stages,
        image_channels,
        patch_embedding_out_channels,
        local_kernel_size,
        patch_size,
        num_attention_heads,
        attention_dropout_prob,
        hidden_dropout_prob,
        exp_factor,
        layer_norm_eps=1e-6,
        use_fusion_layer = True,
    ) -> None:
        super().__init__(patch_embedding_out_channels)
        self.task = task
        self.intermediate_features = self.task in ['segmentation', 'detection']
        
        self.patch_embed = MobileViTEmbeddings(image_channels, patch_embedding_out_channels)
        self.encoder = MobileViTEncoder(config_stages, patch_embedding_out_channels, local_kernel_size, patch_size, num_attention_heads, attention_dropout_prob, hidden_dropout_prob, layer_norm_eps, use_fusion_layer)
        
        encoder_out_channel = config_stages[-1]['out_channels']
        exp_channels = min(exp_factor * encoder_out_channel, 960)
        self.conv_1x1_exp = ConvLayer(opts=None, in_channels=encoder_out_channel, out_channels=exp_channels,
                                      kernel_size=1, stride=1,
                                      use_act=True, use_norm=True)
        self.pool = GlobalPool(pool_type="mean", keep_dim=False)
        
        self._last_channels = exp_channels
        
    def forward(self, x: FXTensorType):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.conv_1x1_exp(x)
        feat = self.pool(x)
        return BackboneOutput(last_feature=feat)

def mobilevit(task, **conf_model):
    mv2_exp_mult = 4
    num_heads = 4
    configuration = {
        "config_stages": [
            {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            {  # 28x28
                "out_channels": 96,
                "hidden_size": 144,
                "intermediate_size": 288,
                "num_transformer_blocks": 2,
                "stride": 2,
                "expand_ratio": mv2_exp_mult,
                "dilate": False,
                "block_type": "mobilevit",
            },
            {  # 14x14
                "out_channels": 128,
                "hidden_size": 192,
                "intermediate_size": 384,
                "num_transformer_blocks": 4,
                "stride": 2,
                "expand_ratio": mv2_exp_mult,
                "dilate": False,
                "block_type": "mobilevit",
            },
            {  # 7x7
                "out_channels": 160,
                "hidden_size": 240,
                "intermediate_size": 480,
                "num_transformer_blocks": 3,
                "stride": 2,
                "expand_ratio": mv2_exp_mult,
                "dilate": False,
                "block_type": "mobilevit",
            }
        ],
        "image_channels": 3,
        "patch_embedding_out_channels": 16,
        "local_kernel_size": 3,
        "patch_size": 2,
        "num_attention_heads": num_heads,
        "attention_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.0,
        "exp_factor": 4,
        "layer_norm_eps": 1e-5,
        "use_fusion_layer": True,
    }
    return MobileViT(task, **configuration)