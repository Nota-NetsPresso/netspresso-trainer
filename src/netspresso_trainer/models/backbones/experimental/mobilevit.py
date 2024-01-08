"""
Based on the mobilevit official implementation.
https://github.com/apple/ml-cvnets/blob/6acab5e446357cc25842a90e0a109d5aeeda002f/cvnets/models/classification/mobilevit.py
"""

import argparse
import math
from typing import Any, Dict, Literal, Optional, Tuple, Union, List

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...op.base_metaformer import ChannelMLP, MetaFormer, MetaFormerBlock, MetaFormerEncoder, MultiHeadAttention
from ...op.custom import ConvLayer, GlobalPool, InvertedResidual
from ...utils import BackboneOutput, FXTensorType

__all__ = ['mobilevit']
SUPPORTING_TASK = ['classification']

class MobileViTEmbeddings(nn.Module):
    def __init__(self, image_channels, hidden_size):
        super().__init__()
        
        image_channels = 3  # {RGB}
        self.conv = ConvLayer(
            in_channels=image_channels,
            out_channels=hidden_size,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
            act_type='silu',
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
            ConvLayer(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=local_kernel_size, stride=1, 
                      use_norm=True, use_act=True, act_type='silu'),
            ConvLayer(in_channels=in_channels, out_channels=hidden_size,
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
        
        self.proj = ConvLayer(in_channels=hidden_size, out_channels=in_channels,
                              kernel_size=1, stride=1,
                              use_norm=True, use_act=True, act_type='silu')
        
        self.use_fusion_layer = use_fusion_layer
        if self.use_fusion_layer:
            self.fusion = ConvLayer(in_channels=in_channels * 2, out_channels=in_channels,
                                    kernel_size=local_kernel_size, stride=1,
                                    use_norm=True, use_act=True, act_type='silu')
        
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
        patches.dim()
        
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
    def __init__(
        self,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        super().__init__()
        stages = []
        
        self.dilation = 1
        self.patch_size = params.patch_size
        self.num_attention_heads = params.num_attention_heads
        self.attention_dropout_prob = params.attention_dropout_prob
        self.hidden_dropout_prob = params.ffn_dropout_prob
        self.use_fusion_layer = params.use_fusion_layer

        # Fix as constant
        self.layer_norm_eps = 1e-5
        self.local_kernel_size = 3

        # Fix as constant
        in_channels = 16
        for stage_param in stage_params:
            stages.append(self._make_block(stage_param, in_channels))
            in_channels = stage_param.out_channels
        self.blocks = nn.Sequential(*stages)
    
    def _make_block(self, stage_param, in_channels):
        out_channels = stage_param.out_channels
        block_type = stage_param.block_type
        stride = stage_param.stride
        num_blocks = stage_param.num_blocks
        expand_ratio = stage_param.ir_expansion_ratio
        
        #out_channels, block_type: Literal['mv2', 'mobilevit'], num_blocks, stride, hidden_size, intermediate_size, num_transformer_blocks, dilate, expand_ratio, in_channels
        if block_type == 'mobilevit':
            num_transformer_blocks = num_blocks
            hidden_size = stage_param.attention_channels
            intermediate_size = stage_param.ffn_intermediate_channels
            dilate = stage_param.dilate

            return self._make_mobilevit_blocks(
                num_transformer_blocks, in_channels, out_channels, stride, expand_ratio,
                hidden_size, intermediate_size, self.patch_size, self.local_kernel_size, self.num_attention_heads,
                self.attention_dropout_prob, self.hidden_dropout_prob, self.layer_norm_eps, self.use_fusion_layer,
                dilate
            )
            
        return self._make_inverted_residual_blocks(
            num_blocks, in_channels, out_channels, stride, expand_ratio
        )
    
    def _make_inverted_residual_blocks(self, num_blocks, in_channels, out_channels, stride, expand_ratio):
        blocks = [
            InvertedResidual(
                in_channels=in_channels if block_idx == 0 else out_channels,
                hidden_channels=in_channels*expand_ratio if block_idx == 0 else out_channels*expand_ratio,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride if block_idx == 0 else 1,
                dilation=1,
                act_type='silu',
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
                    in_channels=in_channels,
                    hidden_channels=in_channels*expand_ratio,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if not dilate else 1,
                    dilation=self.dilation,
                    act_type='silu',
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
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        exp_channels = min(params.output_expansion_ratio * stage_params[-1].out_channels, 960)
        hidden_sizes = [stage.out_channels for stage in stage_params] + [exp_channels]
        super().__init__(hidden_sizes)
        
        self.task = task
        self.intermediate_features = self.task in ['segmentation', 'detection']
        
        image_channels = 3
        # Fix as constant
        patch_embedding_out_channels = 16
        self.patch_embed = MobileViTEmbeddings(image_channels, patch_embedding_out_channels)
        self.encoder = MobileViTEncoder(params=params, stage_params=stage_params)
        
        self.conv_1x1_exp = ConvLayer(in_channels=stage_params[-1].out_channels, out_channels=exp_channels,
                                      kernel_size=1, stride=1,
                                      use_act=True, use_norm=True, act_type='silu')
        self.pool = GlobalPool(pool_type="mean", keep_dim=False)
        
        self._feature_dim = exp_channels
        
    def forward(self, x: FXTensorType):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.conv_1x1_exp(x)
        feat = self.pool(x)
        return BackboneOutput(last_feature=feat)
    
def mobilevit(task, conf_model_backbone):
    return MobileViT(task, conf_model_backbone.params, conf_model_backbone.stage_params)
