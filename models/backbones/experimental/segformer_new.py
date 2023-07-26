from typing import Optional
import math

import torch
import torch.nn as nn

from models.op.depth import DropPath
from models.utils import SeparateForwardModule

from models.configuration.segformer import SegformerConfig
from models.op.base_metaformer import MetaFormer, MetaFormerBlock, MetaFormerEncoder, MultiHeadAttention, ChannelMLP
from models.op.custom import ConvLayer
from models.registry import ACTIVATION_REGISTRY

SUPPORTING_TASK = ['classification', 'segmentation']

TEMP_HIDDEN_SZIE_AS_CONSTANT = 256

class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, in_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, H_embed, W_embed = embeddings.size()
        # (batch_size, in_channels, H_embed, W_embed) -> (batch_size, in_channels, H_embed*W_embed) -> (batch_size, H_embed*W_embed, in_channels)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, H_embed, W_embed

class SegformerDWConv(nn.Module):
    def __init__(self, intermediate_size=768):
        super().__init__()
        self.dwconv = nn.Conv2d(intermediate_size, intermediate_size, 3, 1, 1, bias=True, groups=intermediate_size)

    def forward(self, hidden_states, height, width):
        B, N, C = hidden_states.size()  # N: height*width
        hidden_states = hidden_states.transpose(1, 2).view(B, C, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):
    def __init__(self, in_features, intermediate_size, hidden_dropout_prob, hidden_activation_type):
        super().__init__()
        self.dense1 = nn.Linear(in_features, intermediate_size)
        self.dwconv = SegformerDWConv(intermediate_size)
        self.intermediate_act_fn = ACTIVATION_REGISTRY[hidden_activation_type]()
        self.dense2 = nn.Linear(intermediate_size, in_features)
        
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SegFormerBlock(MetaFormerBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, sequence_reduction_ratio,
                 intermediate_ratio, hidden_dropout_prob, hidden_activation_type,
                 layer_norm_eps=1e-5):
        super().__init__(hidden_size, layer_norm_eps)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = MultiHeadAttention(
            hidden_size, num_attention_heads, attention_dropout_prob,
            use_qkv_bias=True,
            sequence_reduction_ratio=sequence_reduction_ratio
        )
        intermediate_size = int(hidden_size * intermediate_ratio)
        self.channel_mlp = SegformerMixFFN(hidden_size, intermediate_size, hidden_dropout_prob, hidden_activation_type)
        
    def forward(self, x, height, width):
        out_token_mixer = self.layernorm_before(x)
        out_token_mixer = self.token_mixer(out_token_mixer, height=height, width=width)
        
        out_token_mixer = out_token_mixer + x
        
        out_final = self.layernorm_after(out_token_mixer)
        out_final = self.channel_mlp(out_final, height=height, width=width)
        
        out_final = out_final + out_token_mixer
        
        return out_final

class SegformerEncoder(MetaFormerEncoder):
    def __init__(self, use_intermediate_features,
                 image_channels, num_blocks, embedding_patch_sizes, embedding_strides, hidden_sizes,
                 num_attention_heads, attention_dropout_prob, sr_ratios,
                 intermediate_ratio, hidden_dropout_prob, hidden_activation_type, layer_norm_eps):
        super().__init__()
        self.config = SegformerConfig()
        self.use_intermediate_features = use_intermediate_features
        self.num_blocks = num_blocks
        # stochastic depth decay rule
        # drop_path_decays = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, sum(self.config.depths))]

        # patch embeddings
        blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            block = nn.ModuleDict(
                {
                    'segformer_patch_embed': SegformerOverlapPatchEmbeddings(
                        embedding_patch_sizes[i],
                        embedding_strides[i],
                        image_channels if i == 0 else hidden_sizes[i - 1],
                        hidden_sizes[i]
                    ),
                    'segformer_block': SegFormerBlock(
                        hidden_sizes[i],
                        num_attention_heads[i],
                        attention_dropout_prob,
                        sr_ratios[i],
                        intermediate_ratio,
                        hidden_dropout_prob,
                        hidden_activation_type,
                        layer_norm_eps
                    ),
                    'segformer_layer_norm': nn.LayerNorm(hidden_sizes[i])
                 }
            )
            blocks.append(block)
        self.blocks = blocks

    def forward(self, x):
        B = x.size(0)
        all_hidden_states = () if self.use_intermediate_features else None
        
        for block in self.blocks:
            x, H_embed, W_embed = block['segformer_patch_embed'](x)
            x = block['segformer_block'](x, height=H_embed, width=W_embed)
            x = block['segformer_layer_norm'](x)
            
            x = x.reshape(B, H_embed, W_embed, -1).permute(0, 3, 1, 2).contiguous()

            if self.use_intermediate_features:
                all_hidden_states = all_hidden_states + (x,)
        
        if self.use_intermediate_features:
            return all_hidden_states
        return x

class SegFormer(MetaFormer):
    def __init__(self, task,
                 image_channels, num_blocks, embedding_patch_sizes, embedding_strides, hidden_sizes,
                 num_attention_heads, attention_dropout_prob, sr_ratios,
                 intermediate_ratio, hidden_dropout_prob, hidden_activation_type, layer_norm_eps):
        super().__init__(hidden_sizes[-1])
        self.task = task
        self.use_intermediate_features = self.task in ['segmentation', 'detection']
        
        self._last_channels = hidden_sizes[-1]
        
        self.patch_embed = nn.Identity()
        self.encoder = SegformerEncoder(
            self.use_intermediate_features,
            image_channels, num_blocks, embedding_patch_sizes, embedding_strides, hidden_sizes,
            num_attention_heads, attention_dropout_prob, sr_ratios,
            intermediate_ratio, hidden_dropout_prob, hidden_activation_type, layer_norm_eps    
        )
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        out = self.norm(x)
        
        if not self.use_intermediate_features:
            B, C, _, _ = out.size()
            out = out.reshape(B, C, -1)
            feat = torch.mean(out.reshape(B, C, -1), dim=2)
            return {'last_feature': feat}

        return {'intermediate_features': out}
        
def segformer(task, num_class=1000, *args, **kwargs) -> SegformerEncoder:
    configuration = {
        'image_channels': 3,
        'num_blocks': 4,
        'sr_ratios': [8, 4, 2, 1],
        'hidden_sizes': [32, 64, 160, 256],
        'embedding_patch_sizes': [7, 3, 3, 3],
        'embedding_strides': [4, 2, 2, 2],
        'num_attention_heads': [1, 2, 5, 8],
        'intermediate_ratio': 4,
        'hidden_activation_type': "gelu",
        'hidden_dropout_prob': 0.0,
        'attention_dropout_prob': 0.0,
        'layer_norm_eps': 1e-6,
    }
    return SegFormer(task, **configuration)
