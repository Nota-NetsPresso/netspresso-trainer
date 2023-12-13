#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
"""
Based on the YOLOX implementation of Megvii.
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
"""
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ....op.custom import ConvLayer
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

        self.num_classes = num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = ConvLayer

        hidden_dim = int(intermediate_features_dim[0]) # 256 * width
        for i in range(len(intermediate_features_dim)):
            self.stems.append(
                Conv(
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

    def forward(self, xin):
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
