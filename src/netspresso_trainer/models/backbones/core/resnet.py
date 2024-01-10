"""
Based on the Torchvision implementation of ResNet.
https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""
from typing import Dict, List, Literal, Optional, Type, Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import BasicBlock, Bottleneck, ConvLayer
from ...utils import BackboneOutput

__all__ = ['resnet']

SUPPORTING_TASK = ['classification', 'segmentation']

BLOCK_FROM_LITERAL: Dict[str, Type[nn.Module]] = {
    'basicblock': BasicBlock,
    'bottleneck': Bottleneck,
}


class ResNet(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        super(ResNet, self).__init__()

        block: Literal['basicblock', 'bottleneck'] = params.block_type
        norm_layer: Optional[str] = params.norm_type

        # Fix as constant
        zero_init_residual: bool = False
        groups: int = 1
        width_per_group: int = 64
        expansion: Optional[int] = None

        self.task = task.lower()
        block = BLOCK_FROM_LITERAL[block.lower()]
        self.use_intermediate_features = self.task in ['segmentation', 'detection']

        if norm_layer is None:
            norm_layer = 'batch_norm'
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if expansion is None:
            expansion = block.expansion

        self.conv1 = ConvLayer(in_channels=3, out_channels=self.inplanes,
                               kernel_size=7, stride=2, padding=3,
                               bias=False, norm_type='batch_norm', act_type='relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages: List[nn.Module] = []

        first_stage = stage_params[0]
        layer = self._make_layer(block, first_stage.channels, first_stage.num_blocks, expansion=expansion)
        stages.append(layer)
        for stage in stage_params[1:]:
            layer = self._make_layer(block, stage.channels, stage.num_blocks, stride=2,
                                     dilate=stage.replace_stride_with_dilation,
                                     expansion=expansion)
            stages.append(layer)

        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        hidden_sizes = [stage.channels * expansion for stage in stage_params]
        self._feature_dim = hidden_sizes[-1]
        self._intermediate_features_dim = hidden_sizes

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

        all_hidden_states = () if self.use_intermediate_features else None
        for stage in self.stages:
            x = stage(x)
            if self.use_intermediate_features:
                all_hidden_states = all_hidden_states + (x,)

        if self.use_intermediate_features:
            return BackboneOutput(intermediate_features=all_hidden_states)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return BackboneOutput(last_feature=x)

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def resnet(task, conf_model_backbone) -> ResNet:
    """
        ResNet model from "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf.
    """
    return ResNet(task, conf_model_backbone.params, conf_model_backbone.stage_params)
