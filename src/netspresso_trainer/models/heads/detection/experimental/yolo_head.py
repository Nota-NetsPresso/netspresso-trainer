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
Based on the YOLO implementation of WongKinYiu
https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py
"""
import math
import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch import Tensor
from torch.fx.proxy import Proxy
from typing import Dict, List, Optional, Tuple, Union
from ....op.custom import Anchor2Vec, ConvLayer
from ....utils import ModelOutput

def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
    """
        Round up `x` to the biggest-nearest multiple of `div`
    """
    return x + (-x % div)


class Detection(nn.Module):
    """
        A single detection head.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_classes: int,
                 act_type: Optional[str] = None,
                 reg_max: Optional[int] = 16,
                 use_group: bool = True,
                 prior_prob: Optional[float] = 1e-2
        ):
        super().__init__()
        groups = 4 if use_group else 1
        reg_channels = 4 * reg_max
        reg_hidden_channels = max(round_up(hidden_channels // 4, groups), reg_channels, reg_max)
        cls_hidden_channels = max(hidden_channels, min(num_classes * 2, 128))

        self.reg_convs = nn.Sequential(
            ConvLayer(in_channels, reg_hidden_channels, kernel_size=3, act_type=act_type),
            ConvLayer(reg_hidden_channels, reg_hidden_channels, kernel_size=3, groups=groups, act_type=act_type),
            nn.Conv2d(reg_hidden_channels, reg_channels, kernel_size=1, groups=groups)
        )

        self.cls_convs = nn.Sequential(
            ConvLayer(in_channels, cls_hidden_channels, kernel_size=3, act_type=act_type),
            ConvLayer(cls_hidden_channels, cls_hidden_channels, kernel_size=3, act_type=act_type),
            nn.Conv2d(cls_hidden_channels, num_classes, kernel_size=1)
        )

        self.anchor2vec = Anchor2Vec(reg_max=reg_max)

        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)

        bias_reg = self.reg_convs[-1].bias.view(1, -1)
        bias_reg.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.reg_convs[-1].bias = torch.nn.Parameter(bias_reg.view(-1), requires_grad=True)

        bias_cls = self.cls_convs[-1].bias.view(1, -1)
        bias_cls.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.cls_convs[-1].bias = torch.nn.Parameter(bias_cls.view(-1), requires_grad=True)
    
    def forward(self, x: Union[Tensor, Proxy]) -> Tuple[Union[Tensor, Proxy]]:
        reg = self.reg_convs(x)
        cls_logits = self.cls_convs(x)
        output = torch.cat([reg, cls_logits], dim=1)
        return output


class YOLODetectionHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 intermediate_features_dim: List[int],
                 params: DictConfig):
        super().__init__()
        self._validate_params(params)
        self.num_classes = num_classes
        self.num_anchors = params.num_anchors
        self.hidden_dim = int(intermediate_features_dim[0])
        self.heads = self._build_heads(
            intermediate_features_dim,
            params.act_type,
            params.reg_max,
            params.use_group,
        )

        # if params.use_aux_loss:
        #     self.aux_heads = self._build_heads(
        #         intermediate_features_dim,
        #         params.act_type,
        #         params.reg_max,
        #         params.use_group,
        #     )
        # else:
        #     self.aux_heads = None

    def _validate_params(self, params: DictConfig) -> None:
        required_params = ['act_type', 'use_group', 'reg_max', 'num_anchors', 'use_aux_loss']
        for param in required_params:
            if not hasattr(params, param):
                raise ValueError(f"Missing required parameter: {param}")
    
    def _build_heads(
            self,
            intermediate_features_dim: List[int],
            act_type: str,
            reg_max: int,
            use_group: bool,
    ) -> nn.ModuleList:
        heads = nn.ModuleList()
        for feature_dim in intermediate_features_dim:
            head = Detection(
                int(feature_dim),
                self.hidden_dim,
                num_classes=self.num_classes,
                act_type=act_type,
                reg_max=reg_max,
                use_group=use_group,
            )
            heads.append(head)
        return heads

    def forward(self, x_in: Union[List[Tensor], Dict]) -> ModelOutput:
        if isinstance(x_in, Dict):
            assert self.aux_heads
            aux_in = x_in["aux_outputs"]
            x_in = x_in["outputs"]
        else:
            aux_in = None
        outputs = [head(x) for head, x in zip(self.heads, x_in)]
        # if self.training and self.aux_heads:
        #     aux_outputs = [head(x) for head, x in zip(self.aux_heads, aux_in)]
        #     outputs = {"outputs": outputs, "aux_outputs": aux_outputs}
        return ModelOutput(pred=outputs)

def yolo_detection_head(num_classes, intermediate_features_dim, conf_model_head, **kwargs):
    return YOLODetectionHead(num_classes=num_classes,
                             intermediate_features_dim=intermediate_features_dim,
                             params=conf_model_head.params)