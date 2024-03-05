import math
from typing import Optional, List, Dict

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ...op.base_metaformer import Image2Sequence, MetaFormer, MetaFormerBlock, MetaFormerEncoder, MultiHeadAttention
from ...op.custom import ConvLayer
from ...op.registry import ACTIVATION_REGISTRY
from ...utils import BackboneOutput

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

        self.flat = Image2Sequence()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, H_embed, W_embed = embeddings.size()
        # (batch_size, in_channels, H_embed, W_embed) -> (batch_size, in_channels, H_embed*W_embed) -> (batch_size, H_embed*W_embed, in_channels)
        embeddings = self.flat(embeddings)
        embeddings = self.layer_norm(embeddings)
        return embeddings, H_embed, W_embed


class SegformerDWConv(nn.Module):
    def __init__(self, intermediate_size=768):
        super().__init__()
        self.dwconv = nn.Conv2d(intermediate_size, intermediate_size, 3, 1, 1, bias=True, groups=intermediate_size)
        self.flat = Image2Sequence()

    def forward(self, hidden_states, height, width):
        B, N, C = hidden_states.size()  # N: height*width
        hidden_states = hidden_states.transpose(1, 2).view(B, C, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = self.flat(hidden_states)

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
            hidden_size, num_attention_heads,
            attention_scale=(hidden_size // num_attention_heads) ** -0.5,
            attention_dropout_prob=attention_dropout_prob,
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
    def __init__(self, num_blocks, hidden_size,
                 num_attention_heads, attention_dropout_prob, sr_ratio,
                 intermediate_ratio, hidden_dropout_prob, hidden_activation_type, layer_norm_eps):
        super().__init__()
        # stochastic depth decay rule
        # drop_path_decays = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, sum(self.config.depths))]

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                SegFormerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    sr_ratio,
                    intermediate_ratio,
                    hidden_dropout_prob,
                    hidden_activation_type,
                    layer_norm_eps
                )
            )

    def forward(self, x, height, width):
        for block in self.blocks:
            x = block(x, height, width)
        return x


class MixTransformer(MetaFormer):
    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        super().__init__([stage.attention_chananels for stage in stage_params])
        self.task = task
        self.use_intermediate_features = self.task in ['segmentation', 'detection']

        intermediate_ratio = params.ffn_intermediate_expansion_ratio
        hidden_activation_type = params.ffn_act_type
        hidden_dropout_prob = params.ffn_dropout_prob
        attention_dropout_prob = params.attention_dropout_prob

        layer_norm_eps = 1e-5
        in_channels = 3
        
        self.encoder_modules = nn.ModuleList()
        for blocks in stage_params:
            num_blocks = blocks.num_blocks
            sr_ratios = blocks.sequence_reduction_ratio
            hidden_sizes = blocks.attention_chananels
            embedding_patch_sizes = blocks.embedding_patch_sizes
            embedding_strides = blocks.embedding_strides
            num_attention_heads = blocks.num_attention_heads

            module = nn.ModuleDict(
                {
                    'patch_embed': SegformerOverlapPatchEmbeddings(
                        embedding_patch_sizes,
                        embedding_strides,
                        in_channels,
                        hidden_sizes
                    ),
                    'encoder': SegformerEncoder(
                        num_blocks,
                        hidden_sizes,
                        num_attention_heads,
                        attention_dropout_prob,
                        sr_ratios,
                        intermediate_ratio,
                        hidden_dropout_prob,
                        hidden_activation_type,
                        layer_norm_eps
                    ),
                    'norm': nn.LayerNorm(hidden_sizes)
                }
            )
            self.encoder_modules.append(module)

            in_channels = hidden_sizes

    def forward(self, x):
        B = x.size(0)
        all_hidden_states = () if self.use_intermediate_features else None

        for module in self.encoder_modules:
            x, H_embed, W_embed = module['patch_embed'](x)
            x = module['encoder'](x, height=H_embed, width=W_embed)
            x = module['norm'](x)

            x = x.reshape(B, H_embed, W_embed, -1).permute(0, 3, 1, 2).contiguous()

            if self.use_intermediate_features:
                all_hidden_states = all_hidden_states + (x,)

        if self.use_intermediate_features:
            return BackboneOutput(intermediate_features=all_hidden_states)

        B, C, _, _ = x.size()
        feat = torch.mean(x.reshape(B, C, -1), dim=2)
        return BackboneOutput(last_feature=feat)


def mixtransformer(task, conf_model_backbone) -> MixTransformer:
    return MixTransformer(task, conf_model_backbone.params, conf_model_backbone.stage_params)
