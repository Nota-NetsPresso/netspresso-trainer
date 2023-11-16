"""
Based on the publicly available MixNet-PyTorch repository.
https://github.com/romulus0914/MixNet-PyTorch/blob/master/mixnet.py
"""
from typing import List, Dict, Optional

from omegaconf import DictConfig
from torch.nn import functional as F
from collections import OrderedDict
from ...op.registry import ACTIVATION_REGISTRY
from ...op.custom import ConvLayer
Swish = ACTIVATION_REGISTRY['swish']
from ...utils import BackboneOutput

from torch import nn
import torch
import math


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduced_dim, act_type="swish"):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("act", Swish() if act_type == "swish" else nn.ReLU()),
            ("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x, output_size=(1, 1))))
        return torch.mul(x, x_se)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# GPConv: Grouped Point-wise Convolution for MixDepthBlock
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class GPConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_sizes):
        super(GPConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_in_dim = in_planes // self.num_groups
        sub_out_dim = out_planes // self.num_groups

        self.group_point_wise = nn.ModuleList()
        for _ in kernel_sizes:
            self.group_point_wise.append(nn.Conv2d(sub_in_dim, sub_out_dim,
                                                   kernel_size=1, stride=1, padding=0,
                                                   groups=1, dilation=1, bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.group_point_wise[0](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.group_point_wise[stream](chunks[stream]) for stream in range(self.num_groups)]
        return torch.cat(mix, dim=1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MDConv: Mixed Depth-wise Convolution for MixDepthBlock
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MDConv(nn.Module):
    def __init__(self, in_planes, kernel_sizes, stride=1, dilate=1):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_hidden_dim = in_planes // self.num_groups

        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate

        self.mixed_depth_wise = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = ((kernel_size - 1) // 2) * dilate
            self.mixed_depth_wise.append(nn.Conv2d(sub_hidden_dim, sub_hidden_dim,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   groups=sub_hidden_dim, dilation=dilate, bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depth_wise[0](x)

        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.mixed_depth_wise[stream](chunks[stream]) for stream in range(self.num_groups)]
        return torch.cat(mix, dim=1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MixDepthBlock: MixDepthBlock for MixNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MixDepthBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 expand_ratio, exp_kernel_sizes, kernel_sizes, poi_kernel_sizes, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2, act_type="swish"):
        super(MixDepthBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio

        self.groups = len(kernel_sizes)
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = nn.Sequential(OrderedDict([
                ("conv", GPConv(in_planes, hidden_dim, kernel_sizes=exp_kernel_sizes)),
                ("norm", nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)),
                ("act", Swish() if act_type == "swish" else nn.ReLU())
            ]))

        # step 2. Depth-wise convolution phase
        self.depth_wise = nn.Sequential(OrderedDict([
            ("conv", MDConv(hidden_dim, kernel_sizes=kernel_sizes, stride=stride, dilate=dilate)),
            ("norm", nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)),
            ("act", Swish() if act_type == "swish" else nn.ReLU())
        ]))

        # step 3. Squeeze and Excitation
        if self.use_se:
            reduced_dim = max(1, int(in_planes / reduction_ratio))
            self.se_block = SEBlock(hidden_dim, reduced_dim, act_type=act_type)

        # step 4. Point-wise convolution phase
        self.point_wise = nn.Sequential(OrderedDict([
            ("conv", GPConv(hidden_dim, out_planes, kernel_sizes=poi_kernel_sizes)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
        ]))

    def forward(self, x):
        res = x

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

        # step 5. Skip connection and drop connect
        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, )
            x = x + res

        return x


class MixNet(nn.Module):
    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ):
        super(MixNet, self).__init__()

        stem_planes = params.stem_planes
        width_multi = params.width_multi
        depth_multi = params.depth_multi
        self.dropout_rate = params.dropout_rate

        settings = stage_params
        
        out_channels = self._round_filters(stem_planes, width_multi)
        self.mod1 = ConvLayer(in_channels=3, out_channels=out_channels, kernel_size=3,
                              stride=2, groups=1, dilation=1, act_type="relu")

        in_channels = out_channels
        drop_rate = self.dropout_rate
        mod_id = 0
        for t, c, n, k, ek, pk, s, d, a, se in settings:
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)

            if self.dropout_rate:
                drop_rate = self.dropout_rate * float(mod_id + 1) / len(settings)

            # Create blocks for module
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1
                dilate = d if stride == 1 else 1

                blocks.append(("block%d" % (block_id + 1), MixDepthBlock(in_channels, out_channels,
                                                                         expand_ratio=t, exp_kernel_sizes=ek,
                                                                         kernel_sizes=k, poi_kernel_sizes=pk,
                                                                         stride=stride, dilate=dilate,
                                                                         reduction_ratio=se,
                                                                         dropout_rate=drop_rate,
                                                                         act_type=a)))

                in_channels = out_channels
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        self.last_channels = 1536
        self.last_feat = ConvLayer(in_channels=in_channels, out_channels=self.last_channels,
                                   kernel_size=1, stride=1, groups=1, dilation=1, act_type="relu")
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._feature_dim = self.last_channels

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
        return int(self._make_divisible(filters * width_multi))

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))
    
    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def forward(self, x):
        x = self.mod2(self.mod1(x))  # (N, C,   H/2,  W/2)
        x = self.mod4(self.mod3(x))  # (N, C,   H/4,  W/4)
        x = self.mod6(self.mod5(x))  # (N, C,   H/8,  W/8)
        x = self.mod10(self.mod9(self.mod8(self.mod7(x))))  # (N, C,   H/16, W/16)
        x = self.mod12(self.mod11(x))  # (N, C,   H/32, W/32)
        x = self.last_feat(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.training and (self.dropout_rate is not None):
            x = F.dropout(input=x, p=self.dropout_rate,
                          training=self.training, )

        return BackboneOutput(last_feature=x)


def mixnet(task, conf_model_backbone) -> MixNet:
    return MixNet(task, conf_model_backbone.params, conf_model_backbone.stage_params)