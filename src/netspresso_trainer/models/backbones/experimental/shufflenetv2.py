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

from typing import Dict, Optional, List 
from omegaconf import DictConfig
import torch
import torch.nn as nn 

from ...op.custom import ConvLayer, ShuffleV2Block
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST


__all__ = []
SUPPORTING_TASK = ['detection']

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        # Check task compatibility 
        self.task = task.lower() 
        assert self.task in SUPPORTING_TASK, f"ShuffleNetV2 is not supported on {self.task} task now."
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST
        super().__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = self._get_stage_out_channels(params.model_size)
        self._feature_dim = self.stage_out_channels[-1]
        self._intermediate_features_dim = self.stage_out_channels[-2:]
        
        self._build_network()
 
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        c1 = self.stage2(x)
        c2 = self.stage3(c1)
        c3 = self.stage4(c2)
        
        if self.use_intermediate_features:
            return BackboneOutput(intermediate_features=[c2, c3])
        
        x = self.avgpool(c3)
        x = torch.flatten(x, 1)
        return BackboneOutput(last_feature=x)

    def _get_stage_out_channels(self, model_size: str) -> List[int]:
        channels = {
            '0.5x': [-1, 24, 48, 96, 192],
            '1.0x': [-1, 24, 116, 232, 464],
            '1.5x': [-1, 24, 176, 352, 704],
            '2.0x': [-1, 24, 244, 488, 976]
        }
        if model_size not in channels:
            raise ValueError(f"Unsupported model_size: {model_size}. Available options are: {list(channels.keys())}")
        return channels[model_size]

    def _make_stage(self, in_channels: int, out_channels: int, num_layers: int) -> nn.Sequential:
        layers = []
        for i in range(num_layers):
            stride = 2 if i == 0 else 1
            in_ch = in_channels if i == 0 else out_channels // 2
            layers.append(ShuffleV2Block(in_ch, out_channels, 
                                         hidden_channels=out_channels // 2, 
                                         kernel_size=3, stride=stride))
        return nn.Sequential(*layers)
    
    def _build_network(self):
        in_channels = self.stage_out_channels[1]
        self.conv1 = ConvLayer(3, in_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        for i, num_layers in enumerate(self.stage_repeats):
            stage_name = f"stage{i+2}"
            out_channels = self.stage_out_channels[i+2]
            setattr(self, stage_name, self._make_stage(in_channels, out_channels, num_layers))
            in_channels = out_channels
        
        self.avgpool = nn.AdaptiveAvgPool2d(1) if not self.use_intermediate_features else None

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def shufflenetv2(task, conf_model_backbone) -> ShuffleNetV2: 
    return ShuffleNetV2(task, conf_model_backbone.params, conf_model_backbone.stage_params)

