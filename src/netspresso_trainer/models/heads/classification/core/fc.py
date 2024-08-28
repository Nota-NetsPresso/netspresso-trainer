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

from typing import Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from ....utils import ModelOutput
from ....op.registry import ACTIVATION_REGISTRY
from ....op.custom import ConvLayer


class FC(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, params: DictConfig) -> None:
        super(FC, self).__init__()
        dropout_prob = params.dropout_prob
        num_layers = params.num_layers

        assert num_layers >= 1, "num_hidden_layers must be integer larger than 0"

        prev_size = feature_dim
        classifier = []
        for _ in range(num_layers - 1):
            classifier.append(nn.Linear(prev_size, params.intermediate_channels))
            classifier.append(ACTIVATION_REGISTRY[params.act_type]())
            prev_size = params.intermediate_channels
        classifier.append(nn.Dropout(p=dropout_prob))
        classifier.append(nn.Linear(prev_size, num_classes))
        self.classifier = nn.Sequential(*classifier)
        
    def forward(self, x: Union[Tensor, Proxy], targets=None) -> ModelOutput:
        x = self.classifier(x)
        return ModelOutput(pred=x)

class FCConv(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, params: DictConfig) -> None:
        super(FCConv, self).__init__()
        dropout_prob = params.dropout_prob
        num_layers = params.num_layers

        assert num_layers >= 1, "num_hidden_layers must be integer larger than 0"

        prev_size = feature_dim
        classifier = []
        for _ in range(num_layers - 1):
            classifier.append(ConvLayer(prev_size, params.intermediate_channels, 
                                        kernel_size=1, stride=1, padding=0, bias=False, 
                                        norm_type=params.norm_type, act_type=params.act_type))
            prev_size = params.intermediate_channels
        classifier.append(nn.Dropout(p=dropout_prob))
        classifier.append(ConvLayer(prev_size, num_classes,
                                    kernel_size=1, stride=1, padding=0, bias=True,
                                    use_act=False, use_norm=False))

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: Union[Tensor, Proxy], targets=None) -> ModelOutput:
        x = self.classifier(x.unsqueeze(-1).unsqueeze(-1))
        x = torch.flatten(x, 1)
        return ModelOutput(pred=x)

def fc(feature_dim, num_classes, conf_model_head, **kwargs) -> FC:
    return FC(feature_dim=feature_dim, num_classes=num_classes, params=conf_model_head.params)

def fc_conv(feature_dim, num_classes, conf_model_head, **kwargs) -> FCConv:
    return FCConv(feature_dim=feature_dim, num_classes=num_classes, params=conf_model_head.params)