import warnings
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy
from torchvision.ops.misc import SqueezeExcitation as SElayer

from ..op.registry import ACTIVATION_REGISTRY, NORM_REGISTRY


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


@torch.fx.wrap
def tensor_slice(tensor: Tensor, dim, index):
    return tensor.select(dim, index)


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
        x = x + tensor_slice(self.pe, dim=1, index=x.shape[-2])
        
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
