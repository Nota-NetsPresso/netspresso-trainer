# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import argparse
import math
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy
from torchvision.ops.misc import SqueezeExcitation as SElayer

from ..op.registry import ACTIVATION_REGISTRY, NORM_REGISTRY, POOL2D_RESGISTRY


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def auto_pad(kernel_size: Union[int, Tuple[int, int]], dilation: int = 1, **kwargs) -> Tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


class ConvLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: Optional[int] = 1,
        bias: bool = False,
        padding_mode: Optional[str] = 'zeros',
        use_norm: bool = True,
        norm_type: Optional[str] = None,
        use_act: bool = True,
        act_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        if padding is None:
            padding = (
                int((kernel_size[0] - 1) / 2) * dilation[0],
                int((kernel_size[1] - 1) / 2) * dilation[1],
            )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name='conv', module=conv_layer)

        self.norm_name = None
        if use_norm or norm_type is not None:
            _norm_type = norm_type.lower() if norm_type is not None else 'batch_norm'
            if bias:
                warnings.warn("Bias would be ignored in batch normalization", stacklevel=2)
            assert _norm_type in NORM_REGISTRY
            cls_norm = NORM_REGISTRY[_norm_type]
            norm_layer = cls_norm(num_features=out_channels)
            block.add_module(name='norm', module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        if use_act or act_type is not None:
            _act_type = act_type.lower() if act_type is not None else 'relu'
            assert _act_type in ACTIVATION_REGISTRY
            cls_act = ACTIVATION_REGISTRY[_act_type]
            act_layer = cls_act()
            block.add_module(name='act', module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        return self.block(x)

    def __repr__(self):
        return f"{self.block}"


class SeparableConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        bias: bool = False,
        padding_mode: Optional[str] = 'zeros',
        use_norm: bool = True,
        norm_type: Optional[str] = None,
        use_act: bool = True,
        act_type: Optional[str] = None,
        no_out_act: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if act_type is None:
            act_type = 'relu'
        self.depthwise = ConvLayer(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding, groups=in_channels, bias=bias, padding_mode=padding_mode,
                                   use_norm=use_norm, norm_type=norm_type, use_act=use_act, act_type=act_type,)
        self.pointwise = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   use_norm=use_norm, norm_type=norm_type, use_act=False)
        self.final_act = nn.Identity() if no_out_act else ACTIVATION_REGISTRY[act_type]()

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.final_act(x)
        return x

class RepVGGBlock(nn.Module):
    """
    A convolutional block that combines two convolution layers (kernel and point-wise conv).
    This implementation is based on https://github.com/lyuwenyu/RT-DETR/blob/b444daf79cf25f95b740ae71e80fd165e892739a/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py#L35.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 groups: int = 1,
                 act_type: Optional[str] = None,
                 use_identity: Optional[bool]=True,
                 ):
        if act_type is None:
            act_type = 'silu'
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        assert isinstance(act_type, str)
        assert kernel_size == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, groups=groups, use_act=False)
        self.conv2 = ConvLayer(in_channels, out_channels, 1, groups=groups, use_act=False)
        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if use_identity and out_channels == in_channels else None

        assert act_type in ACTIVATION_REGISTRY
        self.act = ACTIVATION_REGISTRY[act_type]()

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        if hasattr(self, 'conv'):
            y = self.conv(x)
            return y

        y = self.conv1(x) + self.conv2(x) + self.rbr_identity(x) if self.rbr_identity else self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.rbr_identity:
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        else:
            kernelid = 0
            biasid = 0

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Union[ConvLayer, nn.BatchNorm2d, nn.Identity]):
        if isinstance(branch, nn.Identity):
            return 0, 0
        if isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
        else:
            assert isinstance(branch, ConvLayer)
            kernel = branch.block.conv.weight
            running_mean = branch.block.norm.running_mean
            running_var = branch.block.norm.running_var
            gamma = branch.block.norm.weight
            beta = branch.block.norm.bias
            eps = branch.block.norm.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class Pool2d(nn.Module):
    def __init__(self,
                 method: str = "max",
                 kernel_size: int = 2,
                 stride: Optional[int] = None,
                 padding: int = 0,
                 **kwargs):
        super().__init__()
        assert method.lower() in POOL2D_RESGISTRY
        self.pool = POOL2D_RESGISTRY[method.lower()](kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        return self.pool(x)


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
        norm_layer: Optional[str] = None,
        expansion: Optional[int] = None,
        act_type: Optional[str] = None,
        no_out_act: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = 'batch_norm'
        if act_type is None:
            act_type = 'relu'
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if expansion is not None:
            self.expansion = expansion
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvLayer(in_channels=inplanes, out_channels=planes,
                               kernel_size=3, stride=stride, dilation=1, padding=1, groups=1,
                               norm_type=norm_layer, act_type=act_type)

        self.conv2 = ConvLayer(in_channels=planes, out_channels=planes,
                               kernel_size=3, stride=1, dilation=1, padding=1, groups=1,
                               norm_type=norm_layer, use_act=False)

        self.downsample = downsample
        self.final_act = nn.Identity() if no_out_act else ACTIVATION_REGISTRY[act_type]()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.final_act(out)

        return out


class Bottleneck(nn.Module):
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
        norm_layer: Optional[str] = None,
        expansion: Optional[int] = None,
        act_type: Optional[str] = None,
        no_out_act: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = 'batch_norm'
        if act_type is None:
            act_type = 'relu'
        width = int(planes * (base_width / 64.)) * groups
        if expansion is not None:
            self.expansion = expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvLayer(in_channels=inplanes, out_channels=width,
                               kernel_size=1, stride=1,
                               norm_type=norm_layer, act_type=act_type)

        self.conv2 = ConvLayer(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, dilation=dilation, padding=dilation, groups=groups,
                               norm_type=norm_layer, act_type=act_type)

        self.conv3 = ConvLayer(in_channels=width, out_channels=planes * self.expansion,
                               kernel_size=1, stride=1,
                               norm_type=norm_layer, use_act=False)

        self.downsample = downsample
        self.final_act = nn.Identity() if no_out_act else ACTIVATION_REGISTRY[act_type]()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.final_act(out)

        return out


class RepNBottleneck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = True,
                 expansion: float = 1.0,
                 depthwise: bool = False,
                 act_type: Optional[str] = None,
                 **kwargs):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = RepVGGBlock(in_channels, hidden_channels, 3, act_type=act_type, **kwargs)
        if depthwise:
            self.conv2 = SeparableConvLayer(hidden_channels,
                                            out_channels,
                                            kernel_size=3, stride=1,
                                            act_type=act_type)
        else:
            self.conv2 = ConvLayer(hidden_channels,
                                   out_channels,
                                   kernel_size=3, stride=1,
                                   act_type=act_type)
        self.shortcut = shortcut

        if shortcut and (in_channels != out_channels):
            self.shortcut = False
            warnings.warn(f"Residual connection disabled: in_channels ({in_channels}) != out_channels ({out_channels})", stacklevel=2)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        norm_type: Optional[str] = None,
        act_type: Optional[str] = None,
        use_se: bool = False,
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []

        # expand
        if hidden_channels != in_channels:
            layers.append(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    norm_type=norm_type,
                    act_type=act_type,
                )
            )

        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(
            ConvLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=hidden_channels,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        if use_se:
            squeeze_channels = make_divisible(hidden_channels // 4, 8)
            layers.append(se_layer(hidden_channels, squeeze_channels))

        # project
        layers.append(
            ConvLayer(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                norm_type=norm_type,
                use_act=False
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = result + input
        return result


class UniversalInvertedResidualBlock(nn.Module):
    # Based on MobileNetV4: https://arxiv.org/pdf/2404.10518
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        extra_dw: bool = False,
        extra_dw_kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        middle_dw: bool = True,
        middle_dw_kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        norm_type: Optional[str] = None,
        act_type: Optional[str] = None,
        use_se: bool = False,
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers: List[nn.Module] = []

        # extra depthwise conv
        if extra_dw:
            assert extra_dw_kernel_size is not None, "if extra_dw is True, extra_dw_kernel_size must be provided."
            layers.append(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=extra_dw_kernel_size,
                    stride=1,
                    groups=in_channels,
                    norm_type=norm_type,
                    use_act=False # No activation for extra depthwise conv
                )
            )

        # expand
        if hidden_channels != in_channels:
            layers.append(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    norm_type=norm_type,
                    act_type=act_type,
                )
            )

        # middle depthwise
        if middle_dw:
            assert middle_dw_kernel_size is not None, "if middle_dw is True, middle_dw_kernel_size must be provided."
            layers.append(
                ConvLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=middle_dw_kernel_size,
                    stride=stride,
                    groups=hidden_channels,
                    norm_type=norm_type,
                    act_type=act_type,
                )
            )

        if use_se:
            squeeze_channels = make_divisible(hidden_channels // 4, 8)
            layers.append(se_layer(hidden_channels, squeeze_channels))

        # project
        layers.append(
            ConvLayer(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                norm_type=norm_type,
                use_act=False # No activation for project conv
            )
        )

        self.block = nn.Sequential(*layers)
        self.apply_layer_scale = False
        if layer_scale is not None:
            self.apply_layer_scale = True
            self.layer_scale = LayerScale2d(out_channels, layer_scale)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.apply_layer_scale:
            result = self.layer_scale(result)
        if self.use_res_connect:
            result = result + input
        return result


class SinusoidalPositionalEncoding(nn.Module):
    """
    This layer adds sinusoidal positional embeddings to a 3D input tensor. The code has been adapted from
    `Pytorch tutorial <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_

    Args:
        d_model (int): dimension of the input tensor
        dropout (Optional[float]): Dropout rate. Default: 0.0
        max_len (Optional[int]): Max. number of patches (or seq. length). Default: 5000
        channels_last (Optional[bool]): Channels dimension is the last in the input tensor

    Shape:
        - Input: :math:`(N, C, P)` or :math:`(N, P, C)` where :math:`N` is the batch size, :math:`C` is the embedding dimension,
            :math:`P` is the number of patches
        - Output: same shape as the input

    """

    def __init__(
        self,
        d_model: int,
        max_len: Optional[int] = 5000,
        channels_last: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:

        position_last = not channels_last

        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        # add dummy batch dimension
        pos_encoding = pos_encoding.unsqueeze(0)  # [1 x C x P_max)

        patch_dim = -2  # patch dimension is second last (N, P, C)
        if position_last:
            pos_encoding = pos_encoding.transpose(
                1, 2
            )  # patch dimension is last (N, C, P)
            patch_dim = -1  # patch dimension is last (N, C, P)

        super().__init__()

        self.patch_dim = patch_dim
        self.register_buffer("pe", pos_encoding)

    def forward_patch_last(
        self, x, indices: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # seq_length should be the last dim
        if indices is None:
            x = x + self.pe[..., : x.shape[-1]]
        else:
            ndim = x.ndim
            repeat_size = [x.shape[0]] + [-1] * (ndim - 1)

            pe = self.pe.expand(repeat_size)
            selected_pe = torch.gather(pe, index=indices, dim=-1)
            x = x + selected_pe
        return x

    def forward_others(
        self, x, indices: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # seq_length should be the second last dim

        # @deepkyu: [fx tracing] Always `indices` is None
        # if indices is None:
        #     x = x + self.pe[..., : x.shape[-2], :]
        # else:
        #     ndim = x.ndim
        #     repeat_size = [x.shape[0]] + [-1] * (ndim - 1)

        #     pe = self.pe.expand(repeat_size)
        #     selected_pe = torch.gather(pe, index=indices, dim=-2)
        #     x = x + selected_pe

        # x = x + self.pe[..., :seq_index, :]
        if not isinstance(x, torch.fx.Proxy):
            self.last_token_num = x.shape[-2]
        x = x + self.pe[..., : self.last_token_num, :]

        return x

    def forward(self, x, indices: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        if self.patch_dim == -1:
            return self.forward_patch_last(x, indices=indices)
        else:
            return self.forward_others(x, indices=indices)

    # def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
    #     return input, 0.0, 0.0

    # def __repr__(self):
    #     return "{}(dropout={})".format(self.__class__.__name__, self.dropout.p)


class GlobalPool(nn.Module):
    """
    This layers applies global pooling over a 4D or 5D input tensor

    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        # if pool_type not in self.pool_types:
        #     logger.error(
        #         "Supported pool types are: {}. Got {}".format(
        #             self.pool_types, pool_type
        #         )
        #     )
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument(
            "--model.layer.global-pool",
            type=str,
            default="mean",
            help="Which global pooling?",
        )
        return parser

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # @deepkyu: [fx tracing] Always x.dim() == 4
        # if x.dim() == 4:
        #     dims = [-2, -1]
        # elif x.dim() == 5:
        #     dims = [-3, -2, -1]
        # else:
        #     raise NotImplementedError("Currently 2D and 3D global pooling supported")

        return self._global_pool(x, dims=(-2, -1))

    # def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
    #     input = self.forward(input)
    #     return input, 0.0, 0.0

    # def __repr__(self):
    #     return "{}(type={})".format(self.__class__.__name__, self.pool_type)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act_type="silu"):
        super().__init__()
        self.conv = ConvLayer(in_channels=in_channels * 4,
                              out_channels=out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              act_type=act_type)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class CSPLayer(nn.Module):
    """
    C3 in yolov5, CSP Bottleneck with 3 convolutions

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        n (int): Number of Bottlenecks. Default: 1
        shortcut (bool): Whether to use shortcut connections. Default: True
        expansion (float): Channel expansion factor. Default: 0.5
        depthwise (bool): Whether to use depthwise separable convolutions. Default: False
        act_type (str): Activation function type. Default: "silu"
        layer_type (str): Type of CSP layer ("csp", "csprep", or "repncsp"). Default: "csp"
    """


    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act_type="silu",
        layer_type: Optional[str] = "csp",
        **kwargs
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        VALID_LAYER_TYPE = ["csp", "csprep", "repncsp"]
        assert layer_type.lower() in VALID_LAYER_TYPE, f"Invalid layer_type: '{layer_type}'. Must be one of {VALID_LAYER_TYPE}"
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = ConvLayer(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1, act_type=act_type)
        self.conv2 = ConvLayer(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=1,
                              stride=1, act_type=act_type)

        block_mapping = {
            "csp": (DarknetBlock, True),
            "csprep": (RepVGGBlock, False),
            "repncsp": (RepNBottleneck, True)
        }

        block, self.concat = block_mapping[layer_type.lower()]

        if self.concat:
            self.conv3 = ConvLayer(in_channels=2 * hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1, act_type=act_type)
        else:
            self.conv3 = ConvLayer(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1, act_type=act_type)

        module_list = [
            block(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                act_type=act_type,
                **kwargs
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1) if self.concat else x_1 + x_2
        return self.conv3(x)


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int=3,
                 expansion: float=1.0,
                 bias: bool= False,
                 use_identity: Optional[bool]=True,
                 act: str="silu"):
        super(CSPRepLayer, self).__init__()
        warnings.warn(
            "CSPRepLayer is deprecated and will be removed in a future version. "
            "Please use CSPLayer with appropriate configuration instead.",
            DeprecationWarning,
            stacklevel=2
        )
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvLayer(in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias, act_type=act)
        self.conv2 = ConvLayer(in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias, act_type=act)
        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act_type=act, use_identity=use_identity) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvLayer(hidden_channels, out_channels, kernel_size=1, stride=1, bias=bias, act_type=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act_type="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=hidden_channels,
                               kernel_size=1, stride=1, act_type=act_type)
        self.m = nn.ModuleList(
            [
                Pool2d(method='max', kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvLayer(in_channels=conv2_channels, out_channels=out_channels,
                               kernel_size=1, stride=1, act_type=act_type)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class ShuffleV2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        assert stride in [1, 2], "Stride must be either 1 or 2"

        self.stride = stride
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.out_channels = out_channels - in_channels

        self.branch_main = self._create_main_branch()
        self.branch_proj = self._create_proj_branch() if stride == 2 else None

    def _create_main_branch(self) -> nn.Sequential:
        return nn.Sequential(
            ConvLayer(self.in_channels, self.hidden_channels, 1, 1, padding=0),
            ConvLayer(self.hidden_channels, self.hidden_channels, self.kernel_size,
                      stride=self.stride, padding=self.padding, groups=self.hidden_channels, use_act=False),
            ConvLayer(self.hidden_channels, self.out_channels, 1, 1, padding=0),
        )

    def _create_proj_branch(self) -> nn.Sequential:
        return nn.Sequential(
            ConvLayer(self.in_channels, self.in_channels, self.kernel_size,
                      self.stride, padding=self.padding, groups=self.in_channels, use_act=False),
            ConvLayer(self.in_channels, self.in_channels, 1, 1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        else:
            return torch.cat((self.branch_proj(x), self.branch_main(x)), 1)

    def channel_shuffle(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b * c // 2, 2, h * w)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, c // 2, h, w)
        return x[0], x[1]


# Newly defined because of slight difference with Bottleneck of custom.py
class DarknetBlock(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act_type="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=hidden_channels,
                                kernel_size=1, stride=1, act_type=act_type)
        if depthwise:
            self.conv2 = SeparableConvLayer(in_channels=hidden_channels, out_channels=out_channels,
                                            kernel_size=3, stride=1, act_type=act_type)
        else:
            self.conv2 = ConvLayer(in_channels=hidden_channels, out_channels=out_channels,
                                    kernel_size=3, stride=1, act_type=act_type)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class LayerScale2d(nn.Module):
    """
        Based on timm implementation.
    """
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class ELAN(nn.Module):
    """
    unified ELAN structure.
    It supports ['basic', 'repncsp'] ELAN structure.
    This implementation is based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 part_channels: int,
                 process_channels: Optional[int] = None,
                 act_type: Optional[str] = None,
                 layer_type: Optional[str] = None,
                 n: Optional[int] = 1,
                 **kwargs
                 ):
        super().__init__()
        VALID_LAYER_TYPE = ["basic", "repncsp"]
        if layer_type is None:
            layer_type = "basic"
        assert layer_type.lower() in VALID_LAYER_TYPE, f"Invalid layer_type: '{layer_type}'. Must be one of {VALID_LAYER_TYPE}"


        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = ConvLayer(in_channels, part_channels, kernel_size=1, act_type=act_type)
        if layer_type.lower() == "basic":
            self.conv2 = ConvLayer(part_channels // 2, process_channels, kernel_size=3, act_type=act_type)
        elif layer_type.lower() == "repncsp":
            self.conv2 = nn.Sequential(
                CSPLayer(part_channels // 2, process_channels, layer_type="repncsp", act_type=act_type, n=n, **kwargs),
                ConvLayer(process_channels, process_channels, kernel_size=3, padding=1, act_type=act_type)
            )

        if layer_type.lower() == "basic":
            self.conv3 = ConvLayer(process_channels, process_channels, kernel_size=3, act_type=act_type)
        elif layer_type.lower() == "repncsp":
            self.conv3 = nn.Sequential(
                CSPLayer(process_channels, process_channels, layer_type="repncsp", act_type=act_type, n=n, **kwargs),
                ConvLayer(process_channels, process_channels, kernel_size=3, padding=1, act_type=act_type)
            )
        self.conv4 = ConvLayer(part_channels + 2 * process_channels, out_channels, kernel_size=1, act_type=act_type)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(torch.cat([x1, x2, x3, x4], dim=1))
        return x5

class AConv(nn.Module):
    """
        Based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py

        A module that combines average pooling and convolution operations.
        Performs average pooling followed by convolution with optional activation.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size for convolution. Default: 3
            stride (int, optional): Stride for convolution. Default: 2
            pool_kernel_size (int, optional): Kernel size for average pooling. Default: 2
            pool_stride (int, optional): Stride for average pooling. Default: 1
            act_type (str, optional): Activation function type. Default: None
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        pool_kernel_size: int = 2,
        pool_stride: int = 1,
        act_type: Optional[str] = None,
    ) -> None:
        """
        Args:
            x (Union[Tensor, Proxy]): Input tensor

        Returns:
            Union[Tensor, Proxy]: Output tensor after pooling and convolution
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            ),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                act_type=act_type
            )
        )

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        return self.layers(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.layers}"


class ADown(nn.Module):
    """
        Based on https://github.com/WongKinYiu/YOLO/blob/b96c8eaec16cfcabbf79947d98d2c575f0a114ad/yolo/model/module.py.
        A module that combines average and max pooling with convolution operations for feature rediction.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): NUmber of output channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_type: Optional[str]= 'silu',
        ) -> None:
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool2d(method="avg", kernel_size=2, stride=1)
        self.conv1 = ConvLayer(half_in_channels, half_out_channels, act_type=act_type, **mid_layer)
        self.max_pool = Pool2d(method="max", **mid_layer, padding=(1, 1))
        self.conv2 = ConvLayer(half_in_channels, half_out_channels, kernel_size=1, act_type=act_type)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class SPPCSPLayer(nn.Module):
    """
        Based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int] = (5, 9, 13),
        expansion: float = 0.5,
        act_type: str = "silu",

    ):
        super().__init__()
        hidden_channels = int(2 * out_channels * expansion)
        self.pre_conv = nn.Sequential(
            ConvLayer(in_channels, hidden_channels, kernel_size=1, act_type=act_type),
            ConvLayer(hidden_channels, hidden_channels, kernel_size=3, act_type=act_type),
            ConvLayer(hidden_channels, hidden_channels, kernel_size=1, act_type=act_type),
        )
        self.short_conv = ConvLayer(in_channels, hidden_channels, kernel_size=1, act_type=act_type)
        self.paddings = [auto_pad(kernel_size) for kernel_size in kernel_sizes]
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding) for kernel_size, padding in zip(kernel_sizes, self.paddings)])
        self.post_conv = nn.Sequential(*[ConvLayer(4 * hidden_channels, hidden_channels, kernel_size=1, act_type=act_type),
                                         ConvLayer(hidden_channels, hidden_channels, kernel_size=3, act_type=act_type)])
        self.merge_conv = ConvLayer(2 * hidden_channels, out_channels, kernel_size=1, act_type=act_type)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        features = [self.pre_conv(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        features = torch.cat(features, dim=1)
        y1 = self.post_conv(features)
        y2 = self.short_conv(x)
        y = torch.cat((y1, y2), dim=1)
        return self.merge_conv(y)


class SPPELAN(nn.Module):
    """
    SPPELAN module cpmprising multiple pooling and convolution layers.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        act_type: Optional[str] = "silu"
    ):
        super(SPPELAN, self).__init__()
        hidden_channels = hidden_channels or out_channels // 2

        self.conv1 = ConvLayer(in_channels, hidden_channels, kernel_size=1, act_type=act_type)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=5, stride=1, padding=auto_pad(kernel_size=5)) for _ in range(3)])
        self.conv5 = ConvLayer(4 * hidden_channels, out_channels, kernel_size=1, act_type=act_type)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class Anchor2Vec(nn.Module):
    """
        This implementation is based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py.
    """
    def __init__(self,
                 reg_max: int=16):
        super().__init__()
        self.reg_max = reg_max
        self.num_predictions = 4 # Number of predictions per anchor
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anchor2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anchor2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple[Tensor, Tensor]: Tuple of (anchor_tensor, vector_tensor)
            where anchor_tensor has shape (batch_size, r, 4, height, width)
            and vector_tensor has shape (batch_size, height, width)
        """
        batch_size, _, height, width = x.shape
        anchor_x = x.view(batch_size, self.num_predictions, self.reg_max, height, width)
        anchor_x = anchor_x.permute(0, 2, 1, 3, 4)
        self.anchor2vec = self.anchor2vec.to(x.device)
        vector_x = self.anchor2vec(anchor_x.softmax(dim=1))[:, 0]
        return anchor_x, vector_x
