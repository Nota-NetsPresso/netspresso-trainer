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
Based on the Torchvision implementation of ResNet.
https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""
from typing import Dict, List, Literal, Optional, Type, Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import BasicBlock, Bottleneck, ConvLayer
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ['resnet']

SUPPORTING_TASK = ['classification', 'segmentation', 'detection', 'pose_estimation']

BLOCK_FROM_LITERAL: Dict[str, Type[nn.Module]] = {
    'basicblock': BasicBlock,
    'bottleneck': Bottleneck,
}


class ResNet(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        # Check task compatibility
        self.task = task.lower()
        assert self.task in SUPPORTING_TASK, f'ResNet is not supported on {self.task} task now.'
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST
        self.return_stage_idx = params.return_stage_idx if params.return_stage_idx else [len(stage_params) - 1]
        self.split_stem_conv = params.split_stem_conv

        super(ResNet, self).__init__()

        block: Literal['basicblock', 'bottleneck'] = params.block_type
        norm_layer: Optional[str] = params.norm_type

        # Fix as constant
        zero_init_residual: bool = False
        groups: int = 1
        width_per_group: int = 64
        expansion: Optional[int] = None

        block = BLOCK_FROM_LITERAL[block.lower()]

        if norm_layer is None:
            norm_layer = 'batch_norm'
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if expansion is None:
            expansion = block.expansion

        if self.split_stem_conv:
            intermediate_stem_inplanes = self.inplanes // 2
            self.conv1 = nn.Sequential(
                ConvLayer(in_channels=3, out_channels=intermediate_stem_inplanes,
                                kernel_size=3, stride=2, padding=1,
                                bias=False, norm_type='batch_norm', act_type='relu'),
                ConvLayer(in_channels=intermediate_stem_inplanes, out_channels=intermediate_stem_inplanes,
                                kernel_size=3, stride=1, padding=1,
                                bias=False, norm_type='batch_norm', act_type='relu'),
                ConvLayer(in_channels=intermediate_stem_inplanes, out_channels=self.inplanes,
                                kernel_size=3, stride=1, padding=1,
                                bias=False, norm_type='batch_norm', act_type='relu'),
            )   
        else:
            self.conv1 = ConvLayer(in_channels=3, out_channels=self.inplanes,
                                kernel_size=7, stride=2, padding=3,
                                bias=False, norm_type='batch_norm', act_type='relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages: List[nn.Module] = []

        first_stage = stage_params[0]
        layer = self._make_layer(block, first_stage.channels, first_stage.num_blocks, 
                                 expansion=expansion, downsample_flag=params.first_stage_shortcut_conv)
        stages.append(layer)
        for stage in stage_params[1:]:
            layer = self._make_layer(block, stage.channels, stage.num_blocks, stride=2,
                                     dilate=stage.replace_stride_with_dilation,
                                     expansion=expansion, downsample_pooling=stage.replace_stride_with_pooling)
            stages.append(layer)

        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        hidden_sizes = [stage.channels * expansion for stage in stage_params]
        self._feature_dim = hidden_sizes[-1]
        self._intermediate_features_dim = [hidden_sizes[i] for i in self.return_stage_idx]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, expansion: Optional[int] = None,
                    downsample_flag: bool = False, downsample_pooling: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if expansion is None:
            expansion = block.expansion
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion or downsample_flag:
            if downsample_pooling:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    ConvLayer(
                        in_channels=self.inplanes, out_channels=planes * expansion,
                        kernel_size=1, stride=1, bias=False,
                        norm_type=norm_layer, use_act=False
                    )
                )
            else:
                downsample = ConvLayer(
                    in_channels=self.inplanes, out_channels=planes * expansion,
                    kernel_size=1, stride=stride, bias=False,
                    norm_type=norm_layer, use_act=False
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        all_hidden_states = () if self.use_intermediate_features else None
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.use_intermediate_features and stage_idx in self.return_stage_idx:
                all_hidden_states = all_hidden_states + (x,)

        if self.use_intermediate_features:
            return BackboneOutput(intermediate_features=all_hidden_states)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return BackboneOutput(last_feature=x)

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def resnet(task, conf_model_backbone) -> ResNet:
    """
        ResNet model from "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf.
    """
    return ResNet(task, conf_model_backbone.params, conf_model_backbone.stage_params)
