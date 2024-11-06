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
from typing import List, Union, Optional, Tuple
from ....op.custom import ConvLayer, Anchor2Vec, ImplicitAdd, ImplicitMul
from ....utils import ModelOutput

def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
    """
    Round up `x` to the bigger-nearest multiple of `div`
    """
    return x + (-x % div)


class ImplicitDetection(nn.Module):
    """
    A single detection head for the yolov7
    """
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_anchors: int = 3,
            **kwargs
    ):
        super().__init__()
        out_channel = num_classes + 5
        out_channels = out_channel * num_anchors
        self.out_conv = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=1, 
                                  **kwargs)
        self.implicit_add = ImplicitAdd(in_channels)
        self.implicit_mul = ImplicitMul(out_channels)

    def forward(self, x: Union[Tensor, Proxy]):
        x = self.implicit_add(x)
        x = self.out_conv(x)
        x = self.implicit_mul(x)

        return x



class Detection(nn.Module):
    """
    A single detetion head for the anchor-free YOLO models
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

        bias_reg = self.reg_convs[-1].conv.bias.view(1, -1)
        bias_reg.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.reg_convs[-1].conv.bias = torch.nn.Parameter(bias_reg.view(-1), requires_grad=True)

        bias_cls = self.cls_convs[-1].conv.bias.view(1, -1)
        bias_cls.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.cls_convs[-1].conv.bias = torch.nn.Parameter(bias_cls.view(-1), requires_grad=True)

    def forward(self, x: Union[Tensor, Proxy]) -> Tuple[Union[Tensor, Proxy]]:
        reg = self.reg_convs(x)
        cls_logits = self.cls_convs(x)
        anchor_x, vector_x = self.anchor2vec(reg)

        return vector_x, anchor_x, cls_logits


class YOLODetectionHead(nn.Module):
    def __init__(
            self,
            num_classes: int,
            intermediate_features_dim: List[int],
            params: DictConfig,
        ):
        super().__init__()
        act_type = params.act_type
        use_group = params.use_group
        reg_max = params.reg_max
        version = params.version
        num_anchors = params.num_anchors
        assert version in ['v9', 'v7'], "The version of head should be either v7 or v9."
        self.num_classes = num_classes
        if version == 'v7':
            assert isinstance(num_anchors, int)
        else:
            num_anchors = None

        self.heads = nn.ModuleList()
        hidden_dim = int(intermediate_features_dim[0])
        for i in range(len(intermediate_features_dim)):
            if version == 'v9':
                self.heads.append(Detection(int(intermediate_features_dim[i]), 
                                            hidden_dim, 
                                            num_classes=self.num_classes, 
                                            act_type=act_type,
                                            reg_max=reg_max,
                                            use_group=use_group))
            else:
                self.heads.append(ImplicitDetection(int(intermediate_features_dim[i]),
                                                    num_classes=num_classes,
                                                    num_anchors=num_anchors))
    
    def forward(self, x_in, targets=None):
        outputs = []
        for idx, x in enumerate(x_in):
            out = self.heads[idx](x)
            outputs.append(out)
        
        return ModelOutput(pred=outputs)
