# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from typing import List, Dict, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import ConvLayer, UniversalInvertedResidualBlock
from ...op.base_metaformer import MobileMultiQueryAttention2D
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ['mobilenetv4']

SUPPORTING_TASK = ['classification', 'segmentation', 'detection', 'pose_estimation']


class FusedIB(nn.Module):
    # Based on MobileNetV4: https://arxiv.org/pdf/2404.10518
    # Only for MobileNetV4
    def __init__(self, in_channel, hidden_channel, out_channel, kernel_size, stride, norm_type, act_type):
        super().__init__()
        self.block = []
        self.block.append(ConvLayer(in_channel, hidden_channel, kernel_size=kernel_size, 
                                    stride=stride, bias=False, norm_type=norm_type, act_type=act_type))
        self.block.append(ConvLayer(hidden_channel, out_channel, kernel_size=1, 
                                    stride=1, bias=False, norm_type=norm_type, use_act=False))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class MobileNetV4(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        # Check task compatibility
        self.task = task.lower()
        assert self.task in SUPPORTING_TASK, f'MobileNetV4 is not supported on {self.task} task now.'
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST

        super(MobileNetV4, self).__init__()

        # Parameters
        stem_out_channel = params.stem_out_channel
        stem_kernel_size = params.stem_kernel_size
        stem_stride = params.stem_stride
        final_conv_out_channel = params.final_conv_out_channel
        final_conv_kernel_size = params.final_conv_kernel_size
        final_conv_stride = params.final_conv_stride
        norm_type = params.norm_type
        act_type = params.act_type
        self.return_stage_idx = params.return_stage_idx if params.return_stage_idx else [len(stage_params) - 1]
        self.layer_scale = params.layer_scale

        # Define model
        self.conv_stem = ConvLayer(3, stem_out_channel, kernel_size=stem_kernel_size, stride=stem_stride,
                                   bias=False, norm_type=norm_type, act_type=act_type)

        stages = []

        prev_channel = stem_out_channel
        for stage_param in stage_params:
            stage = []

            for i, block_info in enumerate(stage_param):
                block_type = block_info[0]
                if block_type == 'conv':
                    in_channels = prev_channel
                    out_channels = block_info[1]
                    kernel_size = block_info[2]
                    stride = block_info[3]

                    stage.append(
                        ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                  bias=False, norm_type=norm_type, act_type=act_type)
                    )

                elif block_type == 'fi':
                    in_channels = prev_channel
                    out_channels = block_info[1]
                    hidden_channels = block_info[2]
                    kernel_size = block_info[3]
                    stride = block_info[4]

                    stage.append(
                        FusedIB(in_channels, hidden_channels, out_channels, kernel_size, stride, norm_type, act_type)
                    )

                elif block_type == 'uir':
                    in_channels = prev_channel
                    out_channels = block_info[1]
                    hidden_channels = block_info[2]
                    extra_dw = block_info[3]
                    extra_kernel_size = block_info[4]
                    middle_dw = block_info[5]
                    middle_kernel_size = block_info[6]
                    stride = block_info[7]

                    stage.append(
                        UniversalInvertedResidualBlock(in_channels, hidden_channels, out_channels, extra_dw, extra_kernel_size, 
                                                       middle_dw, middle_kernel_size, stride, norm_type, act_type, layer_scale=self.layer_scale)
                    )

                elif block_type == 'mmqa':
                    in_channels = prev_channel
                    out_channels = block_info[1]
                    attention_channel = block_info[2]
                    num_attention_heads = block_info[3]
                    quary_pooling_stride = block_info[4]
                    key_val_downsample = block_info[5]
                    key_val_downsample_kernel_size = block_info[6]
                    key_val_downsample_stride = block_info[7]
                    stride = block_info[8]

                    assert in_channels == out_channels, 'MobileMultiQueryAttention2D requires in_channels == out_channels'
                    assert stride == 1, 'MobileMultiQueryAttention2D only supports stride=1'

                    stage.append(
                        MobileMultiQueryAttention2D(in_channels, num_attention_heads, attention_channel,
                                                    use_qkv_bias=False,
                                                    query_pooling_stride=quary_pooling_stride,
                                                    key_val_downsample=key_val_downsample,
                                                    key_val_downsample_kernel_size=key_val_downsample_kernel_size, 
                                                    key_val_downsample_stride=key_val_downsample_stride,
                                                    layer_scale=self.layer_scale)
                    )

                else:
                    raise ValueError(f'Unknown block type: {block_type}')

                prev_channel = out_channels

            stages.append(stage)

        # Add conv on last stage
        final_conv_in_channel = prev_channel
        stages[-1].append(
            ConvLayer(final_conv_in_channel, final_conv_out_channel, kernel_size=final_conv_kernel_size, 
                      stride=final_conv_stride, bias=False, norm_type=norm_type, act_type=act_type)
        )

        # Build stages
        stages = [nn.Sequential(*stage) for stage in stages]
        self.stages = nn.ModuleList(stages)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = final_conv_out_channel
        stage_out_channels = [stage_param[-1][1] for stage_param in stage_params]
        stage_out_channels[-1] = final_conv_out_channel # Replace with final conv out channel
        self._intermediate_features_dim = [stage_out_channels[i] for i in self.return_stage_idx]

    def forward(self, x: Tensor):
        x = self.conv_stem(x)

        all_hidden_states = () if self.use_intermediate_features else None
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.use_intermediate_features and stage_idx in self.return_stage_idx:
                all_hidden_states = all_hidden_states + (x, )

        if self.use_intermediate_features:
            return BackboneOutput(intermediate_features=all_hidden_states)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return BackboneOutput(last_feature=x)
    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def mobilenetv4(task, conf_model_backbone) -> MobileNetV4:
    return MobileNetV4(task, conf_model_backbone.params, conf_model_backbone.stage_params)
