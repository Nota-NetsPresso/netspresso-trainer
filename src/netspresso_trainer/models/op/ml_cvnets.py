"""
Based on the mobilevit official implementation.
https://github.com/apple/ml-cvnets/blob/6acab5e446357cc25842a90e0a109d5aeeda002f/cvnets/models/classification/mobilevit.py
"""
import argparse
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor

# from .base_layer import BaseLayer
# from .normalization_layers import get_normalization_layer
# from .non_linear_layers import get_activation_fn
from .custom import make_divisible


# class GlobalPool(BaseLayer):
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

# class LinearLayer(BaseLayer):
class LinearLayer(nn.Module):
    """
    Applies a linear transformation to the input data

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d

    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        channel_first: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for linear layers",
        )
        parser.add_argument(
            "--model.layer.linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for Linear layers",
        )
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            # if not self.training:
            #     logger.error("Channel-first mode is only supported during inference")
            # if x.dim() != 4:
            #     logger.error("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with torch.no_grad():
                return F.conv2d(
                    input=x,
                    weight=self.weight.clone()
                    .detach()
                    .reshape(self.out_features, self.in_features, 1, 1),
                    bias=self.bias,
                )
        else:
            x = F.linear(x, weight=self.weight, bias=self.bias)
        return x

    # def __repr__(self):
    #     repr_str = (
    #         "{}(in_features={}, out_features={}, bias={}, channel_first={})".format(
    #             self.__class__.__name__,
    #             self.in_features,
    #             self.out_features,
    #             True if self.bias is not None else False,
    #             self.channel_first,
    #         )
    #     )
    #     return repr_str

    # def profile_module(
    #     self, input: Tensor, *args, **kwargs
    # ) -> Tuple[Tensor, float, float]:
    #     out_size = list(input.shape)
    #     out_size[-1] = self.out_features
    #     params = sum([p.numel() for p in self.parameters()])
    #     macs = params
    #     output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
    #     return output, params, macs

class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Defaults to 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. Defaults to 0
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (bool): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``

        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        *args,
        **kwargs
    ) -> None:
        super().__init__(
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


# class ConvLayer(BaseLayer):
class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. When not specified, 
                                               padding is automatically computed based on kernel size 
                                               and dilation rage. Default is ``None``
        groups (Optional[int]): Number of groups in convolution. Default: ``1``
        bias (Optional[bool]): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        act_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        if use_norm:
            norm_type = getattr(opts, "model.normalization.name", "batch_norm")
            if norm_type is not None and norm_type.find("batch") > -1:
                assert not bias, "Do not use bias when using normalization layers."
            elif norm_type is not None and norm_type.find("layer") > -1:
                bias = True
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

        # if in_channels % groups != 0:
        #     logger.error(
        #         "Input channels are not divisible by groups. {}%{} != 0 ".format(
        #             in_channels, groups
        #         )
        #     )
        # if out_channels % groups != 0:
        #     logger.error(
        #         "Output channels are not divisible by groups. {}%{} != 0 ".format(
        #             out_channels, groups
        #         )
        #     )

        block = nn.Sequential()

        conv_layer = Conv2d(
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

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            # norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            momentum = getattr(opts, "model.normalization.momentum", 0.1)
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = (
            getattr(opts, "model.activation.name", "prelu")
            if act_name is None
            else act_name
        )

        if act_type is not None and use_act:
            # neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            # inplace = getattr(opts, "model.activation.inplace", False)
            # act_layer = get_activation_fn(
            #     act_type=act_type,
            #     inplace=inplace,
            #     negative_slope=neg_slope,
            #     num_parameters=out_channels,
            # )
            act_layer = nn.SiLU(inplace=False)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument(
            "--model.layer.conv-init",
            type=str,
            default="kaiming_normal",
            help="Init type for conv layers",
        )
        parser.add_argument(
            "--model.layer.conv-init-std-dev",
            type=float,
            default=None,
            help="Std deviation for conv layers",
        )
        return parser

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ")"
        return repr_str

    # def profile_module(self, input: Tensor) -> (Tensor, float, float):
    #     if input.dim() != 4:
    #         logger.error(
    #             "Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {}".format(
    #                 input.size()
    #             )
    #         )

    #     b, in_c, in_h, in_w = input.size()
    #     assert in_c == self.in_channels, "{}!={}".format(in_c, self.in_channels)

    #     stride_h, stride_w = self.stride
    #     groups = self.groups

    #     out_h = in_h // stride_h
    #     out_w = in_w // stride_w

    #     k_h, k_w = self.kernel_size

    #     # compute MACS
    #     macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
    #     macs /= groups

    #     if self.bias:
    #         macs += self.out_channels * out_h * out_w

    #     # compute parameters
    #     params = sum([p.numel() for p in self.parameters()])

    #     output = torch.zeros(
    #         size=(b, self.out_channels, out_h, out_w),
    #         dtype=input.dtype,
    #         device=input.device,
    #     )
    #     # print(macs)
    #     return output, params, macs



class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    # def profile_module(
    #     self, input: Tensor, *args, **kwargs
    # ) -> Tuple[Tensor, float, float]:
    #     return module_profile(module=self.block, x=input)

    # def __repr__(self) -> str:
    #     return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
    #         self.__class__.__name__,
    #         self.in_channels,
    #         self.out_channels,
    #         self.stride,
    #         self.exp,
    #         self.dilation,
    #         self.use_res_connect,
    #     )
