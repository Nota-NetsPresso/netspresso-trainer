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

from typing import List 
from omegaconf import DictConfig
import torch 
import torch.nn as nn 
import math

from ....op.custom import SeparableConvLayer
from ....utils import ModelOutput 
from .detection import AnchorGenerator


class YOLOFastestHeadV2(nn.Module): 
    def __init__(
        self,
        num_classes: int, 
        intermediate_features_dim: List[int], 
        params: DictConfig) -> None:
        super().__init__()
        anchors = params.anchors
        self.num_anchors = len(anchors[0]) // 2
        hidden_dim = int(intermediate_features_dim[0])
        self.cls_head = YOLOFastestClassificationHead(hidden_dim, self.num_anchors, num_classes)  
        self.reg_head = YOLOFastestRegressionHead(hidden_dim, self.num_anchors) 

    def forward(self, x, target=None):
        cls_logits, objs = self.cls_head(x)
        bbox_regression = self.reg_head(x)
        outputs = list()
        for reg, obj, logits in zip(bbox_regression, objs, cls_logits):
            reg = reg.view(reg.shape[0], self.num_anchors, -1, reg.shape[-2], reg.shape[-1])
            obj = obj.view(obj.shape[0], self.num_anchors, -1, obj.shape[-2], obj.shape[-1])
            logits = logits.repeat(1, self.num_anchors, 1, 1).view(logits.shape[0], self.num_anchors, -1, logits.shape[-2], logits.shape[-1])
            output = torch.cat([reg, obj, logits], 2)
            outputs.append(output)
        return ModelOutput(pred=outputs)

def yolo_fastest_head_v2(num_classes, intermediate_features_dim, conf_model_head) -> YOLOFastestHeadV2:
    return YOLOFastestHeadV2(num_classes=num_classes, 
                             intermediate_features_dim=intermediate_features_dim,
                             params=conf_model_head.params)


class YOLOFastestClassificationHead(nn.Module):
    def __init__(
        self, 
        in_channels,
        num_anchors, 
        num_classes,
        prior_prob = 1e-2,
        num_layers = 2,
        ) -> None:
        super().__init__()
        self.num_classes = num_classes 
        self.num_anchors = num_anchors
        self.layer = nn.ModuleList()
        self.cls_logits = nn.ModuleList()
        self.obj = nn.ModuleList()
        for _ in range(num_layers):
            self.layer.append(
                nn.Sequential(*[
                    SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
                    SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
                ])
            )
            self.cls_logits.append(nn.Conv2d(in_channels, num_classes, 1, 1, 0, bias=True))
            self.obj.append(nn.Conv2d(in_channels, num_anchors, 1, 1, 0, bias=True))
        
        self.initialize_biases(prior_prob=prior_prob)
        
    
    def initialize_biases(self, prior_prob):
        for conv in self.cls_logits:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def forward(self, x): 
        all_cls_logits = [] 
        all_objs = []
        for idx, out in enumerate(x):
            features = self.layer[idx](out)
            cls_logits = features
            objectness = features
            cls_logits = self.cls_logits[idx](cls_logits)
            objectness = self.obj[idx](objectness)
            all_cls_logits.append(cls_logits)
            all_objs.append(objectness)
        return all_cls_logits, all_objs


class YOLOFastestRegressionHead(nn.Module): 
    def __init__(
        self,
        in_channels, 
        num_anchors,
        num_layers=2,
        ) -> None:
        super().__init__()
        self.layer = nn.ModuleList()
        self.bbox_reg = nn.ModuleList()
        for _ in range(num_layers):
            self.layer.append(
                nn.Sequential(*[
                                SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
                                SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True)
                                ]))
            self.bbox_reg.append(nn.Conv2d(in_channels, 4 * num_anchors, 1, 1, 0))
    
    def forward(self, x): 
        all_bbox_regression = []
        for idx, out in enumerate(x):
            features = self.layer[idx](out)
            bbox_regression = features
            bbox_regression = self.bbox_reg[idx](bbox_regression)
            all_bbox_regression.append(bbox_regression)
        return all_bbox_regression

