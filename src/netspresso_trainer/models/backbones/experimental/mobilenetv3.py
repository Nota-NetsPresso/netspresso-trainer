"""
Based on the Torchvision implementation of MobileNetV3.
https://pytorch.org/vision/main/_modules/torchvision/models/mobilenetv3.html
"""
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import ConvLayer, InvertedResidual
from ...utils import BackboneOutput

__all__ = ['mobilenetv3_small']

SUPPORTING_TASK = ['classification', 'segmentation']


def list_depth(block_info):
    if isinstance(block_info[0], list):
        return 1 + list_depth(block_info[0])
    else:
        return 1


class MobileNetV3(nn.Module):

    def __init__(
        self,
        task: str,
        block_info, # [in_channels, kernel, expended_channels, out_channels, use_se, activation, stride, dilation]
        **kwargs
    ) -> None:
        super(MobileNetV3, self).__init__()

        self.task = task.lower()
        self.use_intermediate_features = self.task in ['segmentation', 'detection']
        norm_type = 'batch_norm'
        act_type = 'hard_swish'

        # building first layer
        firstconv_output_channels = block_info[0][0][0]
        self.conv_first = ConvLayer(
            in_channels=3,
            out_channels=firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
        )

        # building inverted residual blocks
        stages: List[nn.Module] = []

        lastconv_input_channels = block_info[-1][-1][3]
        lastconv_output_channels = 6 * lastconv_input_channels
        for stg_idx, stage_info in enumerate(block_info):
            stage: List[nn.Module] = []

            for block in stage_info:
                in_channels = block[0]
                kernel_size = block[1]
                hidden_channels = block[2]
                out_channels = block[3]
                use_se = block[4]
                act_type_b = block[5].lower()
                stride = block[6]
                dilation = block[7]
                stage.append(
                    InvertedResidual(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     norm_type=norm_type,
                                     act_type=act_type_b,
                                     use_se=use_se,
                                     dilation=dilation)
                )
            
            # add last conv
            if stg_idx == len(block_info) - 1:
                stage.append(
                    ConvLayer(in_channels=lastconv_input_channels,
                              out_channels=lastconv_output_channels,
                              kernel_size=1,
                              norm_type=norm_type,
                              act_type=act_type,)
                )

            stage = nn.Sequential(*stage)
            stages.append(stage)
        
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._feature_dim = lastconv_output_channels
        self._intermediate_features_dim = [s[-1].out_channels for s in self.stages[:-1]]
        self._intermediate_features_dim += [lastconv_output_channels]

        # init weights
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
        x = self.conv_first(x)

        all_hidden_states = () if self.use_intermediate_features else None
        for stage in self.stages:
            x = stage(x)
            if self.use_intermediate_features:
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


def mobilenetv3_small(task, conf_model_backbone) -> MobileNetV3:
    return MobileNetV3(task, **conf_model_backbone)
