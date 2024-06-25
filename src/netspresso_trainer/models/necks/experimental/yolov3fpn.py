"""
This code is modified version of mmdetection.
https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/necks/yolo_neck.py
"""
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...op.custom import ConvLayer
from ...utils import BackboneOutput


class YOLOv3FPNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        double_channel: bool,
        kernel_size: int,
        norm_type: str,
        act_type: str,
        depthwise: bool,
        share_fpn_block: bool,
    ) -> None:
        super().__init__()
        self.share_fpn_block = share_fpn_block
        double_out_channels = out_channels * 2 if double_channel else out_channels
        groups = double_out_channels if depthwise else 1

        # shortcut
        if self.share_fpn_block:
            self.conv1 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   norm_type=norm_type, act_type=act_type)
        self.conv2 = ConvLayer(in_channels=out_channels, out_channels=double_out_channels, kernel_size=kernel_size,
                               groups=groups, norm_type=norm_type, act_type=act_type)
        self.conv3 = ConvLayer(in_channels=double_out_channels, out_channels=out_channels, kernel_size=1,
                               norm_type=norm_type, act_type=act_type)
        self.conv4 = ConvLayer(in_channels=out_channels, out_channels=double_out_channels, kernel_size=kernel_size,
                               groups=groups, norm_type=norm_type, act_type=act_type)
        self.conv5 = ConvLayer(in_channels=double_out_channels, out_channels=out_channels, kernel_size=1,
                               norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        if self.share_fpn_block:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv5(x)
        return out


class YOLOv3FPN(nn.Module):

    def __init__(
        self,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()
        self.input_channels = intermediate_features_dim
        self.out_channels = params.out_channels
        self.double_channel = params.double_channel
        self.kernel_size = params.kernel_size
        self.norm_type = params.norm_type
        self.act_type = params.act_type
        self.share_fpn_block = params.share_fpn_block
        self.depthwise = params.depthwise

        self._intermediate_features_dim = self.out_channels

        self.input_channels = self.input_channels[::-1]
        self.out_channels = self.out_channels[::-1]

        self.conv_blocks = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()

        self.build_conv_blocks()
        self.build_fpn_blocks()
    
    def build_conv_blocks(self):
        if self.share_fpn_block:
            for i in range(1, len(self.input_channels)):
                in_c, out_c = self.input_channels[i], self.out_channels[i]
                inter_c = self.out_channels[i - 1]
                self.conv_blocks.append(self.build_1x1_conv(in_channels=inter_c, out_channels=out_c))
        else:
            in_c, out_c = self.input_channels[0], self.out_channels[0]
            self.conv_blocks.append(self.build_1x1_conv(in_channels=in_c, out_channels=out_c))
    
    def build_fpn_blocks(self):
        for i in range(len(self.input_channels)):
            in_c, out_c = self.input_channels[i], self.out_channels[i]
            in_c = in_c if i == 0 else in_c + out_c
            self.fpn_blocks.append(self.build_fpn_block(in_channels=in_c, out_channels=out_c))

    def build_1x1_conv(self, in_channels, out_channels):
        return ConvLayer(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=1,
                         norm_type=self.norm_type,
                         act_type=self.act_type)
    
    def build_fpn_block(self, in_channels, out_channels):
        return YOLOv3FPNBlock(in_channels=in_channels,
                              out_channels=out_channels,
                              double_channel=self.double_channel,
                              kernel_size=self.kernel_size,
                              norm_type=self.norm_type,
                              act_type=self.act_type,
                              depthwise=self.depthwise,
                              share_fpn_block=self.share_fpn_block)

    def forward(self, inputs):
        outputs = []

        feat = inputs[-1] if self.share_fpn_block else self.conv_blocks[0](inputs[-1])
        tmp = self.fpn_blocks[0](feat)
        outputs.append(tmp)

        if self.share_fpn_block: feat = tmp

        for i, x in enumerate(reversed(inputs[:-1])):
            if self.share_fpn_block:
                feat = self.conv_blocks[i](feat)

            # Cat with low-lvl feats
            feat = F.interpolate(feat, scale_factor=2)
            feat = torch.cat((feat, x), 1)

            tmp = self.fpn_blocks[i+1](feat)
            outputs.append(tmp)

            if self.share_fpn_block: feat = tmp                

        return BackboneOutput(intermediate_features=outputs[::-1])
    
    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim


def yolov3fpn(intermediate_features_dim, conf_model_neck, **kwargs):
    return YOLOv3FPN(intermediate_features_dim=intermediate_features_dim, params=conf_model_neck.params)