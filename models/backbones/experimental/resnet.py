"""
Based on the Torchvision implementation of ResNet.
https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""
from typing import Type, Union, List, Optional

import torch
from torch import Tensor
import torch.nn as nn

from models.op.custom import ConvLayer


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


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[str] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = 'batch_norm'
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvLayer(in_channels=inplanes, out_channels=planes,
                               kernel_size=3, stride=stride, dilation=1, padding=1, groups=1,
                               norm_type=norm_layer, act_type='relu')

        self.conv2 = ConvLayer(in_channels=planes, out_channels=planes,
                               kernel_size=3, stride=1, dilation=1, padding=1, groups=1,
                               norm_type=norm_layer, use_act=False)

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[str] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = 'batch_norm'
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvLayer(in_channels=inplanes, out_channels=width,
                               kernel_size=1, stride=1,
                               norm_type=norm_layer, act_type='relu')

        self.conv2 = ConvLayer(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, dilation=dilation, padding=dilation, groups=groups,
                               norm_type=norm_layer, act_type='relu')

        self.conv3 = ConvLayer(in_channels=width, out_channels=planes * self.expansion,
                               kernel_size=1, stride=1,
                               norm_type=norm_layer, use_act=False)

        self.relu3 = nn.ReLU()

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        task: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_class: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[str] = None
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

        self.conv1 = ConvLayer(in_channels=3, out_channels=self.inplanes,
                               kernel_size=7, stride=2, padding=3,
                               bias=False, norm_type='batch_norm', act_type='relu')

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._last_channels = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_class)

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
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvLayer(
                in_channels=self.inplanes, out_channels=planes * block.expansion,
                kernel_size=1, stride=stride, bias=False,
                norm_type=norm_layer, use_act=False
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
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

        return {'last_feature': x}

    @property
    def last_channels(self):
        return self._last_channels

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def resnet50(task, num_class=1000, **extra_params) -> ResNet:
    """
        ResNet-50 model from "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf.
    """
    return ResNet(task, Bottleneck, [3, 4, 6, 3], num_class=num_class, **extra_params)


def resnet101(task, num_class=1000, **extra_params) -> ResNet:
    """
        ResNet-101 model from "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf.
    """
    return ResNet(task, Bottleneck, [3, 4, 23, 3], num_class=num_class, **extra_params)
