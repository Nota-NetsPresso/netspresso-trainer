from typing import Optional, Tuple, List, Any, Union, Dict
import warnings

import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from models.op.swish import Swish


NORM_REGISTRY: Dict[str, nn.Module] = {
    'batch_norm': nn.BatchNorm2d,
    'instance_norm': nn.InstanceNorm2d,
}

ACTIVATION_REGISTRY: Dict[str, nn.Module] = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'leaky_relu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'swish': Swish
}


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
                warnings.warn("Bias would be ignored in batch normalization")
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
            act_layer = cls_act(inplace=False)
            block.add_module(name='act', module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        return self.block(x)

    def __repr__(self):
        return f"{self.block}"
