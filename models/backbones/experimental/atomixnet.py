
from torch.nn import functional as F
from collections import OrderedDict

from torch import nn
import torch
import math

from models.op.swish import Swish
from torch.nn import ReLU


__all__ = ['atomixnet_supernet', 'atomixnet_l', 'atomixnet_m', 'atomixnet_s']

SUPPORTING_TASK = ['classification']
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class SEBlock(nn.Module):
    def __init__(self, in_planes, reduced_dim, act_type=ReLU):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("act", act_type()),
            ("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.sigmoid(self.channel_se(self.avgpool(x)))
        return torch.mul(x, x_se)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 groups=1, dilate=1, act_type=Swish):
        super(ConvBlock, self).__init__()
        assert stride in [1, 2]
        # dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate

        self.conv_block = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilate, groups=groups, bias=False)),
            ("norm", nn.BatchNorm2d(num_features=out_planes,
                                    eps=1e-3, momentum=0.01)),
            ("act", act_type())
        ]))

    def forward(self, x):
        return self.conv_block(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# GPConv: Grouped Point-wise Convolution for MixDepthBlock
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class GPConv_exp(nn.Module):
    def __init__(self, in_planes, out_planes, act_type=Swish):
        super(GPConv_exp, self).__init__()
        self.group_point_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=1, dilation=1, bias=False)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01)),
            ("act", act_type())
        ]))

    def forward(self, x):
        return self.group_point_wise(x)


class GPConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(GPConv, self).__init__()
        self.group_point_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=1, dilation=1, bias=False)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
        ]))

    def forward(self, x):
        return self.group_point_wise(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MDConv: Mixed Depth-wise Convolution for MixDepthBlock
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MDConv(nn.Module):
    def __init__(self, in_planes, kernel_size=1, stride=1, dilate=1, act_type=Swish):
        super(MDConv, self).__init__()
        assert stride in [1, 2]
        # dilate = 1 if stride > 1 else dilate

        padding = ((kernel_size - 1) // 2) * dilate
        self.mixed_depth_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_planes, in_planes,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=in_planes, dilation=dilate, bias=False)),
            ("norm", nn.BatchNorm2d(in_planes, eps=1e-3, momentum=0.01)),
            ("act", act_type())
        ]))

    def forward(self, x):
        return self.mixed_depth_wise(x)


class subMixDepthBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expand_ratio, ksize, stride, dilate, reduction_ratio=4, act_type=Swish, n_group=1, compression=None):
        super(subMixDepthBlock, self).__init__()
        # hidden_dim = in_planes * expand_ratio
        hidden_dim = (in_planes * expand_ratio) // n_group

        self.expand_ratio = expand_ratio
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)

        if compression:
            cmprssn = [value for (key, value) in compression.items() if 'expansion.group_point_wise.conv' in key]
            cmprssn = cmprssn[0] if len(cmprssn) > 0 else 0
            hidden_dim = round(hidden_dim*(1-cmprssn))

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = GPConv_exp(in_planes, hidden_dim, act_type=act_type)

        # step 2. Depth-wise convolution phase
        self.depth_wise = MDConv(hidden_dim, kernel_size=ksize, stride=stride, dilate=dilate, act_type=act_type)

        # step 3. Squeeze and Excitation
        if self.use_se:
            reduced_dim = max(1, int(in_planes / reduction_ratio))
            if compression:
                se_cmprssn = [value for (key, value) in compression.items() if 'se_block' in key]
                se_cmprssn = se_cmprssn[0] if len(se_cmprssn) > 0 else 0
                reduced_dim = round(reduced_dim*(1-se_cmprssn))
            self.se_block = SEBlock(hidden_dim, reduced_dim, act_type=act_type)

        # step 4. Point-wise convolution phase
        self.point_wise = GPConv(hidden_dim, out_planes)

    def forward(self, x):
        # step 1. Expansion phase/Point-wise convolution
        if self.expand_ratio != 1:
            x = self.expansion(x)

        # step 2. Depth-wise convolution phase
        x = self.depth_wise(x)

        # step 3. Squeeze and Excitation
        if self.use_se:
            x = self.se_block(x)

        # step 4. Point-wise convolution phase
        x = self.point_wise(x)

        return x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MixDepthBlock: MixDepthBlock for AtomixNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MixDepthBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 expand_ratio, kernel_sizes, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2, act_type=Swish, compression=None):
        super(MixDepthBlock, self).__init__()

        self.dropout_rate = dropout_rate
        self.num_groups = len(kernel_sizes)
        self.use_residual = (in_planes == out_planes) and stride == 1
        assert stride in [1, 2]
        # dilate = 1 if stride > 1 else dilate
        self.mdblock = nn.ModuleList()
        for i, ksize in enumerate(kernel_sizes):
            dilatation = dilate[i]
            mdblock_compression = {key: value for (key, value) in compression.items() if f"mdblock.{i}" in key} if compression else None
            self.mdblock.append(subMixDepthBlock(in_planes, out_planes, expand_ratio, ksize, stride, dilatation,
                                reduction_ratio=reduction_ratio, act_type=act_type, n_group=self.num_groups, compression=mdblock_compression))

    def forward(self, x):
        res = x

        if self.num_groups == 1:
            x = self.mdblock[0](x)
        else:
            mix = [self.mdblock[stream](x) for stream in range(self.num_groups)]
            x = mix[0]
            # torch.add()
            for i in mix[1:]:
                x += i
            # x = torch.stack(mix).sum(dim=0)

        # step 5. Skip connection and drop connect
        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training)
            x = x + res

        return x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# AtomixNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class AtomixNet(nn.Module):
    def __init__(self, num_classes=1000, arch='supernet', dropout=None):
        super(AtomixNet, self).__init__()

        # compression rate (automatically found via NAS)
        compression = {
            'supernet': None,
            'l': {
                'mod5.block1.mdblock.0.expansion.group_point_wise.conv': 0.042,
                'mod5.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod5.block1.mdblock.1.expansion.group_point_wise.conv': 0.000,
                'mod5.block1.mdblock.1.se_block.channel_se.linear1': 0.000,
                'mod5.block1.mdblock.2.expansion.group_point_wise.conv': 0.000,
                'mod5.block1.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod5.block1.mdblock.3.expansion.group_point_wise.conv': 0.021,
                'mod5.block1.mdblock.3.se_block.channel_se.linear1': 0.062,
                'mod6.block1.mdblock.0.expansion.group_point_wise.conv': 0.183,
                'mod6.block1.mdblock.0.se_block.channel_se.linear1': 0.100,
                'mod6.block1.mdblock.1.expansion.group_point_wise.conv': 0.408,
                'mod6.block1.mdblock.1.se_block.channel_se.linear1': 0.000,
                'mod6.block2.mdblock.0.expansion.group_point_wise.conv': 0.225,
                'mod6.block2.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod6.block2.mdblock.1.expansion.group_point_wise.conv': 0.392,
                'mod6.block2.mdblock.1.se_block.channel_se.linear1': 0.100,
                'mod7.block1.mdblock.0.expansion.group_point_wise.conv': 0.012,
                'mod7.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod7.block1.mdblock.1.expansion.group_point_wise.conv': 0.000,
                'mod7.block1.mdblock.1.se_block.channel_se.linear1': 0.100,
                'mod7.block1.mdblock.2.expansion.group_point_wise.conv': 0.000,
                'mod7.block1.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod8.block1.mdblock.0.expansion.group_point_wise.conv': 0.092,
                'mod8.block1.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod8.block1.mdblock.1.expansion.group_point_wise.conv': 0.183,
                'mod8.block1.mdblock.1.se_block.channel_se.linear1': 0.050,
                'mod8.block1.mdblock.2.expansion.group_point_wise.conv': 0.133,
                'mod8.block1.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block1.mdblock.3.expansion.group_point_wise.conv': 0.333,
                'mod8.block1.mdblock.3.se_block.channel_se.linear1': 0.050,
                'mod8.block2.mdblock.0.expansion.group_point_wise.conv': 0.167,
                'mod8.block2.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod8.block2.mdblock.1.expansion.group_point_wise.conv': 0.283,
                'mod8.block2.mdblock.1.se_block.channel_se.linear1': 0.150,
                'mod8.block2.mdblock.2.expansion.group_point_wise.conv': 0.383,
                'mod8.block2.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block2.mdblock.3.expansion.group_point_wise.conv': 0.408,
                'mod8.block2.mdblock.3.se_block.channel_se.linear1': 0.000,
                'mod8.block3.mdblock.0.expansion.group_point_wise.conv': 0.233,
                'mod8.block3.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod8.block3.mdblock.1.expansion.group_point_wise.conv': 0.358,
                'mod8.block3.mdblock.1.se_block.channel_se.linear1': 0.100,
                'mod8.block3.mdblock.2.expansion.group_point_wise.conv': 0.342,
                'mod8.block3.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod8.block3.mdblock.3.expansion.group_point_wise.conv': 0.475,
                'mod8.block3.mdblock.3.se_block.channel_se.linear1': 0.100,
                'mod9.block1.mdblock.0.expansion.group_point_wise.conv': 0.233,
                'mod9.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod10.block1.mdblock.0.expansion.group_point_wise.conv': 0.317,
                'mod10.block1.mdblock.0.se_block.channel_se.linear1': 0.225,
                'mod10.block1.mdblock.1.expansion.group_point_wise.conv': 0.333,
                'mod10.block1.mdblock.1.se_block.channel_se.linear1': 0.250,
                'mod10.block1.mdblock.2.expansion.group_point_wise.conv': 0.467,
                'mod10.block1.mdblock.2.se_block.channel_se.linear1': 0.250,
                'mod10.block1.mdblock.3.expansion.group_point_wise.conv': 0.483,
                'mod10.block1.mdblock.3.se_block.channel_se.linear1': 0.275,
                'mod11.block1.mdblock.0.expansion.group_point_wise.conv': 0.300,
                'mod11.block1.mdblock.0.se_block.channel_se.linear1': 0.025,
                'mod11.block1.mdblock.1.expansion.group_point_wise.conv': 0.383,
                'mod11.block1.mdblock.1.se_block.channel_se.linear1': 0.075,
                'mod12.block1.mdblock.0.expansion.group_point_wise.conv': 0.017,
                'mod12.block1.mdblock.0.se_block.channel_se.linear1': 0.025,
                'mod12.block1.mdblock.1.expansion.group_point_wise.conv': 0.008,
                'mod12.block1.mdblock.1.se_block.channel_se.linear1': 0.075,
                'mod12.block1.mdblock.2.expansion.group_point_wise.conv': 0.025,
                'mod12.block1.mdblock.2.se_block.channel_se.linear1': 0.025,
                'mod12.block1.mdblock.3.expansion.group_point_wise.conv': 0.000,
                'mod12.block1.mdblock.3.se_block.channel_se.linear1': 0.000,
                'mod13.block1.mdblock.0.expansion.group_point_wise.conv': 0.214,
                'mod13.block1.mdblock.0.se_block.channel_se.linear1': 0.010,
                'mod13.block1.mdblock.1.expansion.group_point_wise.conv': 0.103,
                'mod13.block1.mdblock.1.se_block.channel_se.linear1': 0.000,
                'mod13.block1.mdblock.2.expansion.group_point_wise.conv': 0.092,
                'mod13.block1.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod14.block1.mdblock.0.expansion.group_point_wise.conv': 0.312,
                'mod14.block1.mdblock.0.se_block.channel_se.linear1': 0.030,
                'mod14.block1.mdblock.1.expansion.group_point_wise.conv': 0.203,
                'mod14.block1.mdblock.1.se_block.channel_se.linear1': 0.030,
                'mod14.block1.mdblock.2.expansion.group_point_wise.conv': 0.135,
                'mod14.block1.mdblock.2.se_block.channel_se.linear1': 0.030,
                'mod14.block2.mdblock.0.expansion.group_point_wise.conv': 0.357,
                'mod14.block2.mdblock.0.se_block.channel_se.linear1': 0.010,
                'mod14.block2.mdblock.1.expansion.group_point_wise.conv': 0.289,
                'mod14.block2.mdblock.1.se_block.channel_se.linear1': 0.010,
                'mod14.block2.mdblock.2.expansion.group_point_wise.conv': 0.180,
                'mod14.block2.mdblock.2.se_block.channel_se.linear1': 0.020
            },
            'm': {
                'mod5.block1.mdblock.0.expansion.group_point_wise.conv': 0.062,
                'mod5.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod5.block1.mdblock.1.expansion.group_point_wise.conv': 0.021,
                'mod5.block1.mdblock.1.se_block.channel_se.linear1': 0.000,
                'mod5.block1.mdblock.2.expansion.group_point_wise.conv': 0.042,
                'mod5.block1.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod5.block1.mdblock.3.expansion.group_point_wise.conv': 0.125,
                'mod5.block1.mdblock.3.se_block.channel_se.linear1': 0.062,
                'mod6.block1.mdblock.0.expansion.group_point_wise.conv': 0.433,
                'mod6.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod6.block1.mdblock.1.expansion.group_point_wise.conv': 0.708,
                'mod6.block1.mdblock.1.se_block.channel_se.linear1': 0.150,
                'mod6.block2.mdblock.0.expansion.group_point_wise.conv': 0.500,
                'mod6.block2.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod6.block2.mdblock.1.expansion.group_point_wise.conv': 0.708,
                'mod6.block2.mdblock.1.se_block.channel_se.linear1': 0.100,
                'mod7.block1.mdblock.0.expansion.group_point_wise.conv': 0.012,
                'mod7.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod7.block1.mdblock.1.expansion.group_point_wise.conv': 0.000,
                'mod7.block1.mdblock.1.se_block.channel_se.linear1': 0.000,
                'mod7.block1.mdblock.2.expansion.group_point_wise.conv': 0.038,
                'mod7.block1.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod8.block1.mdblock.0.expansion.group_point_wise.conv': 0.275,
                'mod8.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod8.block1.mdblock.1.expansion.group_point_wise.conv': 0.300,
                'mod8.block1.mdblock.1.se_block.channel_se.linear1': 0.150,
                'mod8.block1.mdblock.2.expansion.group_point_wise.conv': 0.292,
                'mod8.block1.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block1.mdblock.3.expansion.group_point_wise.conv': 0.650,
                'mod8.block1.mdblock.3.se_block.channel_se.linear1': 0.200,
                'mod8.block2.mdblock.0.expansion.group_point_wise.conv': 0.300,
                'mod8.block2.mdblock.0.se_block.channel_se.linear1': 0.100,
                'mod8.block2.mdblock.1.expansion.group_point_wise.conv': 0.417,
                'mod8.block2.mdblock.1.se_block.channel_se.linear1': 0.300,
                'mod8.block2.mdblock.2.expansion.group_point_wise.conv': 0.533,
                'mod8.block2.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block2.mdblock.3.expansion.group_point_wise.conv': 0.617,
                'mod8.block2.mdblock.3.se_block.channel_se.linear1': 0.200,
                'mod8.block3.mdblock.0.expansion.group_point_wise.conv': 0.442,
                'mod8.block3.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod8.block3.mdblock.1.expansion.group_point_wise.conv': 0.533,
                'mod8.block3.mdblock.1.se_block.channel_se.linear1': 0.200,
                'mod8.block3.mdblock.2.expansion.group_point_wise.conv': 0.650,
                'mod8.block3.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block3.mdblock.3.expansion.group_point_wise.conv': 0.708,
                'mod8.block3.mdblock.3.se_block.channel_se.linear1': 0.350,
                'mod9.block1.mdblock.0.expansion.group_point_wise.conv': 0.423,
                'mod9.block1.mdblock.0.se_block.channel_se.linear1': 0.000,
                'mod10.block1.mdblock.0.expansion.group_point_wise.conv': 0.600,
                'mod10.block1.mdblock.0.se_block.channel_se.linear1': 0.250,
                'mod10.block1.mdblock.1.expansion.group_point_wise.conv': 0.567,
                'mod10.block1.mdblock.1.se_block.channel_se.linear1': 0.325,
                'mod10.block1.mdblock.2.expansion.group_point_wise.conv': 0.667,
                'mod10.block1.mdblock.2.se_block.channel_se.linear1': 0.350,
                'mod10.block1.mdblock.3.expansion.group_point_wise.conv': 0.700,
                'mod10.block1.mdblock.3.se_block.channel_se.linear1': 0.250,
                'mod11.block1.mdblock.0.expansion.group_point_wise.conv': 0.575,
                'mod11.block1.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod11.block1.mdblock.1.expansion.group_point_wise.conv': 0.600,
                'mod11.block1.mdblock.1.se_block.channel_se.linear1': 0.200,
                'mod12.block1.mdblock.0.expansion.group_point_wise.conv': 0.025,
                'mod12.block1.mdblock.0.se_block.channel_se.linear1': 0.025,
                'mod12.block1.mdblock.1.expansion.group_point_wise.conv': 0.008,
                'mod12.block1.mdblock.1.se_block.channel_se.linear1': 0.000,
                'mod12.block1.mdblock.2.expansion.group_point_wise.conv': 0.042,
                'mod12.block1.mdblock.2.se_block.channel_se.linear1': 0.025,
                'mod12.block1.mdblock.3.expansion.group_point_wise.conv': 0.000,
                'mod12.block1.mdblock.3.se_block.channel_se.linear1': 0.075,
                'mod13.block1.mdblock.0.expansion.group_point_wise.conv': 0.280,
                'mod13.block1.mdblock.0.se_block.channel_se.linear1': 0.020,
                'mod13.block1.mdblock.1.expansion.group_point_wise.conv': 0.212,
                'mod13.block1.mdblock.1.se_block.channel_se.linear1': 0.010,
                'mod13.block1.mdblock.2.expansion.group_point_wise.conv': 0.220,
                'mod13.block1.mdblock.2.se_block.channel_se.linear1': 0.020,
                'mod14.block1.mdblock.0.expansion.group_point_wise.conv': 0.353,
                'mod14.block1.mdblock.0.se_block.channel_se.linear1': 0.060,
                'mod14.block1.mdblock.1.expansion.group_point_wise.conv': 0.301,
                'mod14.block1.mdblock.1.se_block.channel_se.linear1': 0.070,
                'mod14.block1.mdblock.2.expansion.group_point_wise.conv': 0.271,
                'mod14.block1.mdblock.2.se_block.channel_se.linear1': 0.060,
                'mod14.block2.mdblock.0.expansion.group_point_wise.conv': 0.357,
                'mod14.block2.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod14.block2.mdblock.1.expansion.group_point_wise.conv': 0.406,
                'mod14.block2.mdblock.1.se_block.channel_se.linear1': 0.070,
                'mod14.block2.mdblock.2.expansion.group_point_wise.conv': 0.278,
                'mod14.block2.mdblock.2.se_block.channel_se.linear1': 0.060
            },
            's': {
                'mod5.block1.mdblock.0.expansion.group_point_wise.conv': 0.375,
                'mod5.block1.mdblock.0.se_block.channel_se.linear1': 0.062,
                'mod5.block1.mdblock.1.expansion.group_point_wise.conv': 0.188,
                'mod5.block1.mdblock.1.se_block.channel_se.linear1': 0.125,
                'mod5.block1.mdblock.2.expansion.group_point_wise.conv': 0.292,
                'mod5.block1.mdblock.2.se_block.channel_se.linear1': 0.188,
                'mod5.block1.mdblock.3.expansion.group_point_wise.conv': 0.354,
                'mod5.block1.mdblock.3.se_block.channel_se.linear1': 0.062,
                'mod6.block1.mdblock.0.expansion.group_point_wise.conv': 0.783,
                'mod6.block1.mdblock.0.se_block.channel_se.linear1': 0.150,
                'mod6.block1.mdblock.1.expansion.group_point_wise.conv': 0.892,
                'mod6.block1.mdblock.1.se_block.channel_se.linear1': 0.450,
                'mod6.block2.mdblock.0.expansion.group_point_wise.conv': 0.808,
                'mod6.block2.mdblock.0.se_block.channel_se.linear1': 0.200,
                'mod6.block2.mdblock.1.expansion.group_point_wise.conv': 0.925,
                'mod6.block2.mdblock.1.se_block.channel_se.linear1': 0.350,
                'mod7.block1.mdblock.0.expansion.group_point_wise.conv': 0.075,
                'mod7.block1.mdblock.0.se_block.channel_se.linear1': 0.100,
                'mod7.block1.mdblock.1.expansion.group_point_wise.conv': 0.012,
                'mod7.block1.mdblock.1.se_block.channel_se.linear1': 0.200,
                'mod7.block1.mdblock.2.expansion.group_point_wise.conv': 0.088,
                'mod7.block1.mdblock.2.se_block.channel_se.linear1': 0.000,
                'mod8.block1.mdblock.0.expansion.group_point_wise.conv': 0.500,
                'mod8.block1.mdblock.0.se_block.channel_se.linear1': 0.150,
                'mod8.block1.mdblock.1.expansion.group_point_wise.conv': 0.650,
                'mod8.block1.mdblock.1.se_block.channel_se.linear1': 0.250,
                'mod8.block1.mdblock.2.expansion.group_point_wise.conv': 0.675,
                'mod8.block1.mdblock.2.se_block.channel_se.linear1': 0.350,
                'mod8.block1.mdblock.3.expansion.group_point_wise.conv': 0.858,
                'mod8.block1.mdblock.3.se_block.channel_se.linear1': 0.300,
                'mod8.block2.mdblock.0.expansion.group_point_wise.conv': 0.567,
                'mod8.block2.mdblock.0.se_block.channel_se.linear1': 0.250,
                'mod8.block2.mdblock.1.expansion.group_point_wise.conv': 0.758,
                'mod8.block2.mdblock.1.se_block.channel_se.linear1': 0.400,
                'mod8.block2.mdblock.2.expansion.group_point_wise.conv': 0.858,
                'mod8.block2.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block2.mdblock.3.expansion.group_point_wise.conv': 0.867,
                'mod8.block2.mdblock.3.se_block.channel_se.linear1': 0.300,
                'mod8.block3.mdblock.0.expansion.group_point_wise.conv': 0.683,
                'mod8.block3.mdblock.0.se_block.channel_se.linear1': 0.300,
                'mod8.block3.mdblock.1.expansion.group_point_wise.conv': 0.833,
                'mod8.block3.mdblock.1.se_block.channel_se.linear1': 0.300,
                'mod8.block3.mdblock.2.expansion.group_point_wise.conv': 0.817,
                'mod8.block3.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod8.block3.mdblock.3.expansion.group_point_wise.conv': 0.892,
                'mod8.block3.mdblock.3.se_block.channel_se.linear1': 0.400,
                'mod9.block1.mdblock.0.expansion.group_point_wise.conv': 0.773,
                'mod9.block1.mdblock.0.se_block.channel_se.linear1': 0.050,
                'mod10.block1.mdblock.0.expansion.group_point_wise.conv': 0.750,
                'mod10.block1.mdblock.0.se_block.channel_se.linear1': 0.450,
                'mod10.block1.mdblock.1.expansion.group_point_wise.conv': 0.700,
                'mod10.block1.mdblock.1.se_block.channel_se.linear1': 0.550,
                'mod10.block1.mdblock.2.expansion.group_point_wise.conv': 0.867,
                'mod10.block1.mdblock.2.se_block.channel_se.linear1': 0.400,
                'mod10.block1.mdblock.3.expansion.group_point_wise.conv': 0.883,
                'mod10.block1.mdblock.3.se_block.channel_se.linear1': 0.550,
                'mod11.block1.mdblock.0.expansion.group_point_wise.conv': 0.833,
                'mod11.block1.mdblock.0.se_block.channel_se.linear1': 0.150,
                'mod11.block1.mdblock.1.expansion.group_point_wise.conv': 0.800,
                'mod11.block1.mdblock.1.se_block.channel_se.linear1': 0.350,
                'mod12.block1.mdblock.0.expansion.group_point_wise.conv': 0.308,
                'mod12.block1.mdblock.0.se_block.channel_se.linear1': 0.200,
                'mod12.block1.mdblock.1.expansion.group_point_wise.conv': 0.267,
                'mod12.block1.mdblock.1.se_block.channel_se.linear1': 0.150,
                'mod12.block1.mdblock.2.expansion.group_point_wise.conv': 0.275,
                'mod12.block1.mdblock.2.se_block.channel_se.linear1': 0.225,
                'mod12.block1.mdblock.3.expansion.group_point_wise.conv': 0.150,
                'mod12.block1.mdblock.3.se_block.channel_se.linear1': 0.125,
                'mod13.block1.mdblock.0.expansion.group_point_wise.conv': 0.790,
                'mod13.block1.mdblock.0.se_block.channel_se.linear1': 0.110,
                'mod13.block1.mdblock.1.expansion.group_point_wise.conv': 0.659,
                'mod13.block1.mdblock.1.se_block.channel_se.linear1': 0.120,
                'mod13.block1.mdblock.2.expansion.group_point_wise.conv': 0.672,
                'mod13.block1.mdblock.2.se_block.channel_se.linear1': 0.100,
                'mod14.block1.mdblock.0.expansion.group_point_wise.conv': 0.865,
                'mod14.block1.mdblock.0.se_block.channel_se.linear1': 0.190,
                'mod14.block1.mdblock.1.expansion.group_point_wise.conv': 0.782,
                'mod14.block1.mdblock.1.se_block.channel_se.linear1': 0.220,
                'mod14.block1.mdblock.2.expansion.group_point_wise.conv': 0.748,
                'mod14.block1.mdblock.2.se_block.channel_se.linear1': 0.190,
                'mod14.block2.mdblock.0.expansion.group_point_wise.conv': 0.801,
                'mod14.block2.mdblock.0.se_block.channel_se.linear1': 0.200,
                'mod14.block2.mdblock.1.expansion.group_point_wise.conv': 0.887,
                'mod14.block2.mdblock.1.se_block.channel_se.linear1': 0.200,
                'mod14.block2.mdblock.2.expansion.group_point_wise.conv': 0.831,
                'mod14.block2.mdblock.2.se_block.channel_se.linear1': 0.210
            }
        }
        compression = compression[arch]

        params = (24, [
            # t, c,  n, k,            s,  d,             a,     se
            [1, 24,  1, [3],          1, [1],          ReLU,   None],  # [2]
            [6, 32,  1, [3, 5],       2, [1, 1],       ReLU,   None],  # [3]
            [3, 32,  1, [3],          1, [1],          ReLU,   None],  # [4]
            [6, 40,  1, [3, 5, 7, 9], 2, [1, 1, 1, 1], Swish,  2],    # [5]
            [6, 40,  2, [3, 5],       1, [1, 1],       Swish,  2],    # [6]
            [6, 80,  1, [3, 5, 7],    2, [1, 1, 1],    Swish,  4],    # [7]
            [6, 80,  3, [3, 5, 7, 9], 1, [1, 1, 1, 1], Swish,  4],    # [8]
            [6, 80,  1, [3],          1, [1],          Swish,  2],    # [9]
            [3, 80,  1, [3, 5, 7, 9], 1, [1, 1, 1, 1], Swish,  2],    # [10]
            [3, 80,  1, [3, 5],       1, [1, 1, 1, 1], Swish,  2],    # [11]
            [6, 200, 1, [3, 5, 7, 9], 2, [1, 1, 1, 1], Swish,  2],    # [12]
            [8, 200, 1, [3, 5, 7],    1, [1, 1, 1],    Swish,  2],    # [13]
            [4, 200, 2, [3, 5, 7],    1, [1, 1, 1],    Swish,  2]     # [14]
        ], 1.0, 1.0, 0.2)

        stem_planes, settings, width_multi, depth_multi, self.dropout_rate = params
        if dropout is not None:
            # replace default dropout if a value was manually specified
            self.dropout_rate = dropout
        out_channels = self._round_filters(stem_planes, width_multi)
        self.mod1 = ConvBlock(3, out_channels, kernel_size=3, stride=2,
                              groups=1, dilate=1, act_type=ReLU)

        in_channels = out_channels
        drop_rate = self.dropout_rate

        mod_id = 2
        for t, c, n, k, s, d, a, se in settings:
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)

            if self.dropout_rate:
                drop_rate = self.dropout_rate * float(mod_id-1) / len(settings)

            # Create blocks for module
            mod_compression = {key: value for (key, value) in compression.items() if f"mod{mod_id}" in key} if compression else None
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1
                dilate = d
                block_compression = {key: value for (key, value) in mod_compression.items() if f"block{block_id + 1}" in key} if compression else None
                blocks.append((f"block{block_id + 1}", MixDepthBlock(in_channels, out_channels,
                                                                     expand_ratio=t,
                                                                     kernel_sizes=k,
                                                                     stride=stride, dilate=dilate,
                                                                     reduction_ratio=se,
                                                                     dropout_rate=drop_rate,
                                                                     act_type=a,
                                                                     compression=block_compression)))

                in_channels = out_channels
            self.add_module(f"mod{mod_id}", nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        self._last_channels = 1536
        self.last_feat = ConvBlock(in_channels, self._last_channels,
                                   kernel_size=1, stride=1,
                                   groups=1, dilate=1, act_type=ReLU)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(self._last_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_divisible(value, divisor=8):
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters, width_multi):
        if width_multi == 1.0:
            return filters
        if isinstance(filters, int):
            return int(self._make_divisible(filters * width_multi))
        return [int(self._make_divisible(f * width_multi)) for f in filters]

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))

    def forward(self, x):
        for module_key in self._modules:
            if 'mod' in module_key:
                x = self._modules[module_key](x)
        x = self.last_feat(x)
        x = self.avgpool(x).reshape(-1, x.size(1))
        if self.training and (self.dropout_rate is not None):
            x = F.dropout(input=x, p=self.dropout_rate, training=self.training)
        # x = self.classifier(x)
        return x

    @property
    def last_channels(self):
        return self._last_channels

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def atomixnet_supernet(num_class=1000, **extra_params):
    return AtomixNet(num_classes=num_class, **extra_params)


def atomixnet_l(num_class=1000, **extra_params):
    return AtomixNet(num_classes=num_class, arch='l', **extra_params)


def atomixnet_m(num_class=1000, **extra_params):
    return AtomixNet(num_classes=num_class, arch='m', **extra_params)


def atomixnet_s(num_class=1000, **extra_params):
    return AtomixNet(num_classes=num_class, arch='s', **extra_params)
