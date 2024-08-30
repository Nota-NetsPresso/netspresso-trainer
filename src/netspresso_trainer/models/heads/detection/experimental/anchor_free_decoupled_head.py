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
Based on the YOLOX implementation of Megvii.
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
"""
from typing import List
import math

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ....op.custom import ConvLayer, SeparableConvLayer
from ....utils import ModelOutput


class AnchorFreeDecoupledHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()
        act_type = params.act_type
        depthwise = params.depthwise

        self.num_classes = num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = SeparableConvLayer if depthwise else ConvLayer

        hidden_dim = int(intermediate_features_dim[0]) # 256 * width
        for i in range(len(intermediate_features_dim)):
            self.stems.append(
                ConvLayer(
                    in_channels=int(intermediate_features_dim[i]),
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    act_type=act_type,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act_type=act_type,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act_type=act_type,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act_type=act_type,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            stride=1,
                            act_type=act_type,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)
        self.initialize_biases(1e-2)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, targets=None):
        outputs = []

        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)

            outputs.append(output)

        return ModelOutput(pred=outputs)


def anchor_free_decoupled_head(num_classes, intermediate_features_dim, conf_model_head, **kwargs) -> AnchorFreeDecoupledHead:
    return AnchorFreeDecoupledHead(num_classes=num_classes,
                                   intermediate_features_dim=intermediate_features_dim,
                                   params=conf_model_head.params)
