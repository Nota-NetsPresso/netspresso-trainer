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

from ...op.custom import ConvLayer, UniversalInvertedBottleneckBlock
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ['mobilenetv4']

SUPPORTING_TASK = ['classification', 'segmentation', 'detection', 'pose_estimation']


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

        # Implement on conv-small
        norm_type = 'batch_norm'
        act_type = 'relu'

        self.conv_stem = ConvLayer(3, 32, kernel_size=3, stride=2, bias=False, norm_type=norm_type, act_type=act_type)

        stages = []

        # TODO: Replace with for loop
        stage1 = [
            ConvLayer(32, 32, kernel_size=3, stride=2, bias=False, norm_type=norm_type, act_type=act_type),
            ConvLayer(32, 32, kernel_size=1, stride=1, bias=False, norm_type=norm_type, act_type=act_type),
        ]

        stage2 = [
            ConvLayer(32, 96, kernel_size=3, stride=2, bias=False, norm_type=norm_type, act_type=act_type),
            ConvLayer(96, 64, kernel_size=1, stride=1, bias=False, norm_type=norm_type, act_type=act_type),
        ]

        stage3 = [
            UniversalInvertedBottleneckBlock(64, 192, 96, True, 5, True, 5, 2, norm_type, act_type),
            UniversalInvertedBottleneckBlock(96, 192, 96, False, None, True, 3, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(96, 192, 96, False, None, True, 3, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(96, 192, 96, False, None, True, 3, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(96, 192, 96, False, None, True, 3, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(96, 384, 96, True, 3, False, None, 1, norm_type, act_type),
        ]

        stage4 = [
            UniversalInvertedBottleneckBlock(96, 576, 128, True, 3, True, 3, 2, norm_type, act_type),
            UniversalInvertedBottleneckBlock(128, 512, 128, True, 5, True, 5, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(128, 512, 128, False, None, True, 5, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(128, 384, 128, False, None, True, 5, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(128, 512, 128, False, None, True, 3, 1, norm_type, act_type),
            UniversalInvertedBottleneckBlock(128, 512, 128, False, None, True, 3, 1, norm_type, act_type),
        ]

        stages = [stage1, stage2, stage3, stage4]

        # Add conv on last stage
        stages[-1].append(ConvLayer(128, 960, kernel_size=1, stride=1, bias=False, norm_type=norm_type, act_type=act_type))

        # Build stages
        stages = [nn.Sequential(*stage) for stage in stages]
        self.stages = nn.ModuleList(stages)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = 960

        # Dummy
        self._intermediate_features_dim = [64, 96, 960]

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
