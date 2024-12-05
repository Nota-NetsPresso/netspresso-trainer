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

from ...op.custom import ConvLayer, ELAN, AConv
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

        out_features=("stage3", "stage4", "stage5")
        assert out_features, "please provide output features of RepNCSP-ELAN"
        self.out_features = out_features
        dep_mul = params.dep_mul
        wid_mul = params.wid_mul
        act_type = params.act_type

        base_channels = int(wid_mul * 64) # t: 0.25, s: 0.5
        base_depth = max(round(dep_mul * 3), 1) # t: 0.33, s: 0.33

        self.stem = ConvLayer(3, base_channels, kernel_size=3, stride=2, act_type=act_type)

        self.stage2 = nn.Sequential(
            ConvLayer(in_channels=base_channels,
                      out_channels=base_channels * 2,
                      kernel_size=3,
                      stride=2,
                      act_type=act_type),
            ELAN(in_channels=base_channels*2,
                 out_channels=base_channels*2,
                 part_channels=base_channels*2,
                 act_type=act_type,
                 use_identity=False,
                 layer_type="basic"
                 )
        )

        self.stage3 = nn.Sequential(
            AConv(in_channels=base_channels*2,
                  out_channels=base_channels*4,
                  act_type=act_type),
            ELAN(in_channels=base_channels*4,
                 out_channels=base_channels*4,
                 part_channels=base_channels*4,
                 act_type=act_type,
                 layer_type="repncsp",
                 use_identity=False,
                 n=base_depth * 3)
        )

        self.stage4 = nn.Sequential(
            AConv(in_channels=base_channels*4,
                  out_channels=base_channels*6,
                  act_type=act_type),
            ELAN(in_channels=base_channels*6,
                 out_channels=base_channels*6,
                 part_channels=base_channels*6,
                 act_type=act_type,
                 layer_type="repncsp",
                 use_identity=False,
                 n=base_depth * 3)
        )

        self.stage5 = nn.Sequential(
            AConv(in_channels=base_channels*6,
                  out_channels=base_channels*8,
                  act_type=act_type),
            ELAN(in_channels=base_channels*8,
                 out_channels=base_channels*8,
                 part_channels=base_channels*8,
                 act_type=act_type,
                 layer_type="repncsp",
                 use_identity=False,
                 n=base_depth * 3)    
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        predefined_out_features = {'stage2': base_channels * 2, 'stage3': base_channels * 4,
                                   'stage4': base_channels * 6, 'stage5': base_channels * 8}
        self._feature_dim = predefined_out_features['stage5']
        self._intermediate_features_dim = [predefined_out_features[out_feature] for out_feature in out_features]

        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)

    
    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        outputs_dict = {}
        x = self.stem(x)
        outputs_dict["stem"] = x
        x = self.stage2(x)
        outputs_dict["stage2"] = x
        x = self.stage3(x)
        outputs_dict["stage3"] = x
        x = self.stage4(x)
        outputs_dict["stage4"] = x
        x = self.stage5(x)
        outputs_dict["stage5"] = x

        if self.use_intermediate_features:
            all_hidden_states = [outputs_dict[out_name] for out_name in self.out_features]
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