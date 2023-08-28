"""
Based on the Torchvision implementation of ResNet.
https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""
from typing import Type, Union, List, Optional

import torch
from torch import Tensor
import torch.nn as nn

from ...op.custom import ConvLayer, BasicBlock, Bottleneck
from ...utils import BackboneOutput

__all__ = ['resnet50', 'resnet101']

SUPPORTING_TASK = ['classification']


# 'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
# 'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth'
# 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
# 'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
# 'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth'
# 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'
# 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'
# 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'
# 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'


class ResNet(nn.Module):

    def __init__(
        self,
        task: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[str] = None,
        expansion: Optional[int] = None,
    ) -> None:
        super(ResNet, self).__init__()

        self.task = task.lower()
        self.intermediate_features = self.task in ['segmentation', 'detection']

        if norm_layer is None:
            norm_layer = 'batch_norm'
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if expansion is None:
            expansion = block.expansion

        self.conv1 = ConvLayer(in_channels=3, out_channels=self.inplanes,
                               kernel_size=7, stride=2, padding=3,
                               bias=False, norm_type='batch_norm', act_type='relu')

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], expansion=expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       expansion=expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       expansion=expansion)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       expansion=expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._last_channels = 512 * expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, expansion: Optional[int] = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if expansion is None:
            expansion = block.expansion
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = ConvLayer(
                in_channels=self.inplanes, out_channels=planes * expansion,
                kernel_size=1, stride=stride, bias=False,
                norm_type=norm_layer, use_act=False
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return BackboneOutput(last_feature=x)

    @property
    def last_channels(self):
        return self._last_channels

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def resnet50(task, **conf_model) -> ResNet:
    """
        ResNet-50 model from "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf.
    """
    configuration = {
        'block': Bottleneck,
        'layers': [3, 4, 6, 3]
    }
    return ResNet(task, **configuration)


# def resnet101(task, **conf_model) -> ResNet:
#     """
#         ResNet-101 model from "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf.
#     """
#     configuration = {
#         'block': Bottleneck,
#         'layers': [3, 4, 23, 3]
#     }
#     return ResNet(task, **configuration)
