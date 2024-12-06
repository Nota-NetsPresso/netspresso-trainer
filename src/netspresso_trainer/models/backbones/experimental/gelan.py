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
from typing import Dict, Optional, List, Union

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from torch.fx.proxy import Proxy

from ...op.custom import ConvLayer, ELAN, AConv, ADown, SPPELAN
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ["gelan"]
SUPPORTING_TASK = ['classification', 'detection']


class GELAN(nn.Module):
    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        self.task = task.lower()
        assert self.task in SUPPORTING_TASK, f"RepNCSP-ELAN is not supported on {self.task} task for now."
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST
        super().__init__()

        # Parameters
        stem_out_channels = params.stem_out_channels
        stem_kernel_size = params.stem_kernel_size
        stem_stride = params.stem_stride
        self.return_stage_idx = params.return_stage_idx if params.return_stage_idx else [len(stage_params) - 1]
        act_type = params.act_type.lower()
        stages = []

        self.conv_stem = ConvLayer(3, stem_out_channels, kernel_size=stem_kernel_size, stride=stem_stride,
                                   act_type=act_type)
        stages = []
        prev_channels = stem_out_channels

        for stage_param in stage_params:
            stage = []
            for idx, block_info in enumerate(stage_param):
                block_type = block_info[0].lower()
                in_channels = prev_channels
                out_channels = block_info[1]
                if block_type == "conv":
                    kernel_size = block_info[2]
                    stride = block_info[3]
                    block = ConvLayer(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, act_type=act_type)
                elif block_type == "elan":
                    part_channels = block_info[2]
                    use_identity = block_info[3]
                    block = ELAN(in_channels=in_channels, out_channels=out_channels, part_channels=part_channels,
                                 use_identity=use_identity, act_type=act_type, layer_type="basic")
                elif block_type == "sppelan":
                    hidden_channels = block_info[2]
                    block = SPPELAN(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                    act_type=act_type)
                elif block_type == "repncspelan":
                    part_channels = block_info[2]
                    use_identity = block_info[3]
                    depth = block_info[4]
                    block = ELAN(in_channels=in_channels, out_channels=out_channels, part_channels=part_channels,
                                 use_identity=use_identity, act_type=act_type, layer_type="repncsp", n=depth)
                elif block_type == "aconv":
                    block = AConv(in_channels=in_channels, out_channels=out_channels, act_type=act_type)
                elif block_type == "adown":
                    block = ADown(in_channels=in_channels, out_channels=out_channels, act_type=act_type)
                else:
                    raise ValueError(f'Unknown block type: {block_type}')
                prev_channels = out_channels
                stage.append(block)
                assert len(stage) == idx + 1
            
            stages.append(stage)
        
        stages = [nn.Sequential(*stage) for stage in stages]
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        stage_out_channels = [stage_param[-1][1] for stage_param in stage_params]
        self._intermediate_features_dim = [stage_out_channels[i] for i in self.return_stage_idx]
        self._feature_dim = stage_out_channels[-1]
        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)

    
    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
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


def gelan(task, conf_model_backbone) -> GELAN:
    return GELAN(task, conf_model_backbone.params, conf_model_backbone.stage_params)