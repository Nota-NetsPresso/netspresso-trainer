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

"""
Based on the Torchvision implementation of MobileNetV3.
https://pytorch.org/vision/main/_modules/torchvision/models/mobilenetv3.html
"""
from typing import List, Dict, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import ConvLayer, InvertedResidual
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ['mobilenetv3']

SUPPORTING_TASK = ['classification', 'segmentation', 'detection', 'pose_estimation']


class MobileNetV3(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        # Check task compatibility
        self.task = task.lower()
        assert self.task in SUPPORTING_TASK, f'MobileNetV3 is not supported on {self.task} task now.'
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST

        super(MobileNetV3, self).__init__()

        norm_type = 'batch_norm'
        act_type = 'hard_swish'

        # building first layer
        firstconv_output_channels = stage_params[0].in_channels[0]
        self.conv_first = ConvLayer(
            in_channels=3,
            out_channels=firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
        )

        # building inverted residual blocks
        stages: List[nn.Module] = []

        lastconv_input_channels = stage_params[-1].out_channels[-1]
        lastconv_output_channels = 6 * lastconv_input_channels
        for stg_idx, stage_info in enumerate(stage_params):
            stage: List[nn.Module] = []

            for block in zip(stage_info.in_channels, stage_info.kernel_sizes, stage_info.expanded_channels,
                             stage_info.out_channels, stage_info.use_se, stage_info.act_type,
                             stage_info.stride):
                in_channels, kernel_size, hidden_channels, out_channels, use_se, act_type_b, stride = block
                dilation = 1 # Fix as a constant
                act_type_b = act_type_b.lower()
                stage.append(
                    InvertedResidual(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     norm_type=norm_type,
                                     act_type=act_type_b,
                                     use_se=use_se,
                                     dilation=dilation)
                )
            
            # add last conv
            if stg_idx == len(stage_params) - 1:
                stage.append(
                    ConvLayer(in_channels=lastconv_input_channels,
                              out_channels=lastconv_output_channels,
                              kernel_size=1,
                              norm_type=norm_type,
                              act_type=act_type,)
                )

            stage = nn.Sequential(*stage)
            stages.append(stage)
        
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._feature_dim = lastconv_output_channels
        self._intermediate_features_dim = [s[-1].out_channels for s in self.stages[:-1]]
        self._intermediate_features_dim += [lastconv_output_channels]

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # Set batchnorm eps and momentum
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 0.001
                    m.momentum = 0.01
        self.apply(init_bn)

    def forward(self, x: Tensor):
        x = self.conv_first(x)

        all_hidden_states = () if self.use_intermediate_features else None
        for stage in self.stages:
            x = stage(x)
            if self.use_intermediate_features:
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


def mobilenetv3(task, conf_model_backbone) -> MobileNetV3:
    return MobileNetV3(task, conf_model_backbone.params, conf_model_backbone.stage_params)
