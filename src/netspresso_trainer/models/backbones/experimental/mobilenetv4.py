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
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ['mobilenetv4']

SUPPORTING_TASK = ['classification', 'segmentation', 'detection', 'pose_estimation']


class FusedIB(nn.Module):
    # Based on MobileNetV4: https://arxiv.org/pdf/2404.10518
    # Only for MobileNetV4
    def __init__(self, in_channel, hidden_channel, out_channel, kernel_size, stride, norm_type, act_type, out_act):
        super().__init__()
        self.block = []
        self.block.append(ConvLayer(in_channel, hidden_channel, kernel_size=kernel_size, 
                                    stride=stride, bias=False, norm_type=norm_type, act_type=act_type))
        self.block.append(ConvLayer(hidden_channel, out_channel, kernel_size=1, 
                                    stride=1, bias=False, norm_type=norm_type, use_act=out_act, act_type=act_type if out_act else None))
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

        # Define model
        self.conv_stem = ConvLayer(3, stem_out_channel, kernel_size=stem_kernel_size, stride=stem_stride,
                                   bias=False, norm_type=norm_type, act_type=act_type)

        stages = []

        for stage_param in stage_params:
            stage = []

            block_type = stage_param['block_type']
            if block_type == 'fused_inverted':
                in_channels = stage_param['in_channels']
                hidden_channels = stage_param['hidden_channels']
                out_channels = stage_param['out_channels']
                kernel_sizes = stage_param['kernel_size']
                strides = stage_param['stride']
                out_acts = stage_param['out_act']

                block_info = zip(in_channels, hidden_channels, out_channels, kernel_sizes, strides, out_acts)
                for in_channel, hidden_channel, out_channel, kernel_size, stride, out_act in block_info:
                    stage.append(
                        FusedIB(in_channel, hidden_channel, out_channel, kernel_size, stride, norm_type, act_type, out_act)
                    )

            elif block_type == 'universal_inverted_residual':
                in_channels = stage_param['in_channels']
                hidden_channels = stage_param['hidden_channels']
                out_channels = stage_param['out_channels']
                extra_dws = stage_param['extra_dw']
                extra_kernel_sizes = stage_param['extra_dw_kernel_size']
                middle_dws = stage_param['middle_dw']
                middle_kernel_sizes = stage_param['middle_dw_kernel_size']
                strides = stage_param['stride']

                block_info = zip(in_channels, hidden_channels, out_channels, extra_dws, extra_kernel_sizes, middle_dws, middle_kernel_sizes, strides)
                for in_channel, hidden_channel, out_channel, extra_dw, extra_kernel_size, middle_dw, middle_kernel_size, stride in block_info:
                    stage.append(
                        UniversalInvertedResidualBlock(in_channel, hidden_channel, out_channel, extra_dw, extra_kernel_size, middle_dw, middle_kernel_size, stride, norm_type, act_type)
                    )

            else:
                raise ValueError(f'Unknown block type: {block_type}')

            stages.append(stage)

        # Add conv on last stage
        final_conv_in_channel = stage_params[-1]['out_channels'][-1]
        stages[-1].append(
            ConvLayer(final_conv_in_channel, final_conv_out_channel, kernel_size=final_conv_kernel_size, 
                      stride=final_conv_stride, bias=False, norm_type=norm_type, act_type=act_type)
        )

        # Build stages
        stages = [nn.Sequential(*stage) for stage in stages]
        self.stages = nn.ModuleList(stages)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = final_conv_out_channel
        stage_out_channels = [stage_param['out_channels'][-1] for stage_param in stage_params]
        stage_out_channels[-1] = final_conv_out_channel # Replace with final conv out channel
        self._intermediate_features_dim = [stage_out_channels[i] for i in self.return_stage_idx]

    def forward(self, x: Tensor):
        x = self.conv_stem(x)
        for stage in self.stages:
            x = stage(x)

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
