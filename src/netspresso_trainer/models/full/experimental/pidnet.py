# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import time
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...op.custom import BasicBlock, Bottleneck, ConvLayer
from ...op.pidnet import DAPPM, PAPPM, Bag, Light_Bag, PagFM, segmenthead
from ...utils import FXTensorType, PIDNetModelOutput

use_align_corners = False


class PIDNet(nn.Module):

    def __init__(
            self, 
            params: Optional[Dict] = None
    ) -> None:
        super(PIDNet, self).__init__()
        num_classes = params.num_classes
        m = params.m
        n = params.n
        planes = params.channels
        ppm_planes = params.ppm_channels
        head_planes = params.head_channels
        is_training = params.is_training

        self.is_training = is_training

        # I Branch
        self.conv1 = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=planes,
                      kernel_size=3, stride=2, padding=1,
                      norm_type='batch_norm', act_type='relu'),
            ConvLayer(in_channels=planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1,
                      norm_type='batch_norm', act_type='relu')
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2, expansion=2)

        # P Branch
        self.compression3 = ConvLayer(in_channels=planes * 4, out_channels=planes * 2,
                                      kernel_size=1, stride=1, padding=0,
                                      norm_type='batch_norm', use_act=False)

        self.compression4 = ConvLayer(in_channels=planes * 8, out_channels=planes * 2,
                                      kernel_size=1, stride=1, padding=0,
                                      norm_type='batch_norm', use_act=False)

        self.pag3 = PagFM(planes * 2, planes, resize_to=(512 // 8, 512 // 8))
        self.pag4 = PagFM(planes * 2, planes, resize_to=(512 // 8, 512 // 8))

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1, expansion=2)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1, expansion=2)
            self.diff3 = ConvLayer(in_channels=planes * 4, out_channels=planes,
                                   kernel_size=3, stride=1, padding=1,
                                   norm_type='batch_norm', use_act=False)
            self.diff4 = ConvLayer(in_channels=planes * 8, out_channels=planes * 2,
                                   kernel_size=3, stride=1, padding=1,
                                   norm_type='batch_norm', use_act=False)
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = ConvLayer(in_channels=planes * 4, out_channels=planes * 2,
                                   kernel_size=3, stride=1, padding=1,
                                   norm_type='batch_norm', use_act=False)
            self.diff4 = ConvLayer(in_channels=planes * 8, out_channels=planes * 2,
                                   kernel_size=3, stride=1, padding=1,
                                   norm_type='batch_norm', use_act=False)
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1, expansion=2)

        # Prediction Head
        if self.is_training:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.original_to = (512, 512)
        self.resize_to = (512 // 8, 512 // 8)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, expansion=None):
        downsample = None
        if expansion is None:
            expansion = block.expansion
        if stride != 1 or inplanes != planes * expansion:
            downsample = ConvLayer(in_channels=inplanes, out_channels=planes * expansion,
                                   kernel_size=1, stride=stride,
                                   norm_type='batch_norm', use_act=False)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, expansion=expansion))
        inplanes = planes * expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, expansion=expansion, no_out_act=True))
            else:
                layers.append(block(inplanes, planes, stride=1, expansion=expansion, no_out_act=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1, expansion=None):
        downsample = None
        if expansion is None:
            expansion = block.expansion
        if stride != 1 or inplanes != planes * expansion:
            downsample = ConvLayer(in_channels=inplanes, out_channels=planes * expansion,
                                   kernel_size=1, stride=stride,
                                   norm_type='batch_norm', use_act=False)

        layer = block(inplanes, planes, stride, downsample, expansion=expansion, no_out_act=True)

        return layer

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: FXTensorType, label_size=None) -> PIDNetModelOutput:

        # assert H == x.size(2)
        # assert W == x.size(3)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))

        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=self.resize_to,
            mode='bilinear', align_corners=use_align_corners)

        if self.is_training:
            # if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))

        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=self.resize_to,
            mode='bilinear', align_corners=use_align_corners)

        if self.is_training:
            # if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))

        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=self.resize_to,
            mode='bilinear', align_corners=use_align_corners)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        x_ = F.interpolate(x_, size=self.original_to, mode='bilinear', align_corners=True)

        # if self.augment:
        x_extra_p = self.seghead_p(temp_p)
        x_extra_d = self.seghead_d(temp_d)

        x_extra_p = F.interpolate(x_extra_p, size=self.original_to, mode='bilinear', align_corners=True)
        x_extra_d = F.interpolate(x_extra_d, size=self.original_to, mode='bilinear', align_corners=True)

        return PIDNetModelOutput(extra_p=x_extra_p, extra_d=x_extra_d, pred=x_)

def pidnet(num_classes: int, conf_model_full) -> PIDNet:
    # PIDNet-S
    conf_model_full.num_classes = num_classes
    conf_model_full.is_training = True
    return PIDNet(params=conf_model_full)