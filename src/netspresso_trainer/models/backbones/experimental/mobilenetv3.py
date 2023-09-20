from typing import Dict, List, Literal, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import ConvLayer, InvertedResidual
from ...utils import BackboneOutput
from ...op.registry import ACTIVATION_REGISTRY

__all__ = ['mobilenetv3_small']

SUPPORTING_TASK = ['classification', 'segmentation']


class MobileNetV3(nn.Module):

    def __init__(
        self,
        task: str,
        block_info, # [in_channels, kernel, expended_channels, out_channels, use_se, activation, stride, dilation]
        last_channel,
        **kwargs
    ) -> None:
        super(MobileNetV3, self).__init__()

        self.task = task.lower()
        block = InvertedResidual
        self.use_intermediate_features = self.task in ['segmentation', 'detection']
        norm_type = 'batch_norm'
        act_type = 'hard_swish'

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = block_info[0][0]
        layers.append(
            ConvLayer(
                in_channels=3,
                out_channels=firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                act_type=act_type,
            )
        )

        # building inverted residual blocks
        for cnf in block_info:
            in_channels = cnf[0]
            kernel_size = cnf[1]
            hidden_channels = cnf[2]
            out_channels = cnf[3]
            use_se = cnf[4]
            act_type_b = cnf[5].lower()
            stride = cnf[6]
            dilation = cnf[7]
            layers.append(block(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     norm_type=norm_type,
                                     act_type=act_type_b,
                                     use_se=use_se,
                                     dilation=dilation))

        # building last several layers
        lastconv_input_channels = block_info[-1][3]
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvLayer(
                in_channels=lastconv_input_channels,
                out_channels=lastconv_output_channels,
                kernel_size=1,
                norm_type=norm_type,
                act_type=act_type,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.last_layer = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            ACTIVATION_REGISTRY[act_type](),
        )

        self._feature_dim = last_channel

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last_layer(x)

        return BackboneOutput(last_feature=x)

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return None

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def mobilenetv3_small(task, conf_model_backbone) -> MobileNetV3:
    return MobileNetV3(task, **conf_model_backbone)
