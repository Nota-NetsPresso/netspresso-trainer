import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy
from torchvision.ops.misc import SqueezeExcitation as SElayer

from ..op.ml_cvnets import make_divisible
from ..op.registry import ACTIVATION_REGISTRY, NORM_REGISTRY


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
        no_relu: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = 'batch_norm'
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if expansion is not None:
            self.expansion = expansion
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = ConvLayer(in_channels=inplanes, out_channels=planes,
                               kernel_size=3, stride=stride, dilation=1, padding=1, groups=1,
                               norm_type=norm_layer, act_type='relu')

        self.conv2 = ConvLayer(in_channels=planes, out_channels=planes,
                               kernel_size=3, stride=1, dilation=1, padding=1, groups=1,
                               norm_type=norm_layer, use_act=False)

        self.downsample = downsample
        self.final_act = nn.Identity() if no_relu else nn.ReLU()

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
        no_relu: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = 'batch_norm'
        width = int(planes * (base_width / 64.)) * groups
        if expansion is not None:
            self.expansion = expansion
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

        self.downsample = downsample
        self.final_act = nn.Identity() if no_relu else nn.ReLU()

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
