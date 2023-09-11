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

@torch.fx.wrap
def tensor_slice(tensor: Tensor, dim, index):
    return tensor.select(dim, index)


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
    
class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (Optional[int]): Number of transformer blocks. Default: 2
        head_dim (Optional[int]): Head dimension in the multi-head attention. Default: 32
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: Optional[int] = 2,
        head_dim: Optional[int] = 32,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[int] = 0.0,
        ffn_dropout: Optional[int] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        transformer_norm_layer: Optional[str] = "layer_norm",
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        no_fusion: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        conv_3x3_in = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
        )
        conv_1x1_in = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        conv_1x1_out = ConvLayer(
            opts=opts,
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                opts=opts,
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                stride=1,
                use_norm=True,
                use_act=True,
            )
        super().__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                opts=opts,
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=transformer_norm_layer,
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(
            # get_normalization_layer(
            #     opts=opts,
            #     norm_type=transformer_norm_layer,
            #     num_features=transformer_dim,
            # )
            LayerNorm(
                normalized_shape=transformer_dim,
            )
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(
                feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    def forward_spatial(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm

    def forward_temporal(
        self, x: Tensor, x_prev: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        res = x
        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for global_layer in self.global_rep:
            if isinstance(global_layer, TransformerEncoder):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm, patches

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            # for spatio-temporal MobileViT
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            # For image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    # def profile_module(
    #     self, input: Tensor, *args, **kwargs
    # ) -> Tuple[Tensor, float, float]:
    #     params = macs = 0.0

    #     res = input
    #     out, p, m = module_profile(module=self.local_rep, x=input)
    #     params += p
    #     macs += m

    #     patches, info_dict = self.unfolding(feature_map=out)

    #     patches, p, m = module_profile(module=self.global_rep, x=patches)
    #     params += p
    #     macs += m

    #     fm = self.folding(patches=patches, info_dict=info_dict)

    #     out, p, m = module_profile(module=self.conv_proj, x=fm)
    #     params += p
    #     macs += m

    #     if self.fusion is not None:
    #         out, p, m = module_profile(
    #             module=self.fusion, x=torch.cat((out, res), dim=1)
    #         )
    #         params += p
    #         macs += m

    #     return res, params, macs
    
class LayerNorm(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same shape as the input
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        n_dim = x.ndim
        if x.shape[1] == self.normalized_shape[0] and n_dim > 2:  # channel-first format
            s, u = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - u) / (s + self.eps)
            if self.weight is not None:
                # Using fused operation for performing affine transformation: x = (x * weight) + bias
                n_dim = x.ndim - 2
                new_shape = [1, self.normalized_shape[0]] + [1] * n_dim
                x = torch.addcmul(
                    input=self.bias.reshape(*[new_shape]),
                    value=1.0,
                    tensor1=x,
                    tensor2=self.weight.reshape(*[new_shape]),
                )
            return x
        elif x.shape[-1] == self.normalized_shape[0]:  # channel-last format
            return super().forward(x)
        else:
            raise NotImplementedError(
                "LayerNorm is supported for channel-first and channel-last format only"
            )
            
class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (Optional[int]) : Number of heads in multi-head attention. Default: 8
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        transformer_norm_layer (Optional[str]): Normalization layer. Default: layer_norm

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
                bias=True,
                coreml_compatible=getattr(
                    opts, "common.enable_coreml_compatible_module", False
                ),
            )

        self.pre_norm_mha = nn.Sequential(
            # get_normalization_layer(
            #     opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            # ),
            LayerNorm(
                normalized_shape=embed_dim,
            ),
            attn_unit,
            nn.Dropout(p=dropout),
        )

        act_name = self.build_act_layer(opts=opts)
        self.pre_norm_ffn = nn.Sequential(
            # get_normalization_layer(
            #     opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            # ),
            LayerNorm(
                normalized_shape=embed_dim,
            ),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            nn.Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout),
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        # act_type = getattr(opts, "model.activation.name", "relu")
        # neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        # inplace = getattr(opts, "model.activation.inplace", False)
        # act_layer = get_activation_fn(
        #     act_type=act_type,
        #     inplace=inplace,
        #     negative_slope=neg_slope,
        #     num_parameters=1,
        # )
        act_layer = nn.SiLU(inplace=False)
        return act_layer

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            **kwargs
        )  # mha
        x = self.pre_norm_mha[2](x)  # dropout
        x = x + res

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    # def profile_module(
    #     self, input: Tensor, *args, **kwargs
    # ) -> Tuple[Tensor, float, float]:
    #     b_sz, seq_len = input.shape[:2]

    #     out, p_mha, m_mha = module_profile(module=self.pre_norm_mha, x=input)

    #     out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)
    #     m_ffn = m_ffn * b_sz * seq_len

    #     macs = m_mha + m_ffn
    #     params = p_mha + p_ffn

    #     return input, params, macs
    
class SingleHeadAttention(nn.Module):
    """
    This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim**-0.5

    def forward(
        self,
        x: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        # [N, P, C] --> [N, P, 3C]
        if x_kv is None:
            qkv = self.qkv_proj(x)
            # [N, P, 3C] --> [N, P, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            query = F.linear(
                x,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, C, P]
        key = key.transpose(-2, -1)

        # QK^T
        # [N, P, C] x [N, C, P] --> [N, P, P]
        attn = torch.matmul(query, key)

        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == list(
                attn.shape
            ), "Shape of attention mask and attn should be the same. Got: {} and {}".format(
                attn_mask.shape, attn.shape
            )
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, P]
            batch_size, num_src_tokens, num_tgt_tokens = attn.shape
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                float("-inf"),
            )

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, P, P] x [N, P, C] --> [N, P, C]
        out = torch.matmul(attn, value)
        out = self.out_proj(out)

        return out

    # def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
    #     b_sz, seq_len, in_channels = input.shape
    #     params = macs = 0.0

    #     qkv, p, m = module_profile(module=self.qkv_proj, x=input)
    #     params += p
    #     macs += m * seq_len * b_sz

    #     # number of operations in QK^T
    #     m_qk = (seq_len * in_channels * in_channels) * b_sz
    #     macs += m_qk

    #     # number of operations in computing weighted sum
    #     m_wt = (seq_len * in_channels * in_channels) * b_sz
    #     macs += m_wt

    #     out_p, p, m = module_profile(module=self.out_proj, x=input)
    #     params += p
    #     macs += m * seq_len * b_sz

    #     return input, params, macs


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        # if embed_dim % num_heads != 0:
        #     logger.error(
        #         "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
        #             self.__class__.__name__, embed_dim, num_heads
        #         )
        #     )

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_separate_proj_weight = embed_dim != output_dim

    # def forward_tracing(
    #     self,
    #     x_q: Tensor,
    #     x_kv: Optional[Tensor] = None,
    #     key_padding_mask: Optional[Tensor] = None,
    #     attn_mask: Optional[Tensor] = None,
    # ) -> Tensor:
    #     # Let S: sequence length (H'*W' + 1)
    #     # Let C: C_split * {head}(=num_heads)
    #     # x_q: B x S x C
    #     if x_kv is None:
    #         qkv = self.qkv_proj(x_q)  # B x S x 3C
    #         query, key, value = torch.chunk(qkv, chunks=3, dim=-1)  # B x S x C, B x S x C, B x S x C
    #     else:
    #         query = F.linear(
    #             x_q,
    #             weight=self.qkv_proj.weight[: self.embed_dim, ...],
    #             bias=self.qkv_proj.bias[: self.embed_dim]
    #             if self.qkv_proj.bias is not None
    #             else None,
    #         )  # B x S x C

    #         kv = F.linear(
    #             x_kv,
    #             weight=self.qkv_proj.weight[self.embed_dim :, ...],
    #             bias=self.qkv_proj.bias[self.embed_dim :]
    #             if self.qkv_proj.bias is not None
    #             else None,
    #         )  #  B x S x 2C_h
    #         key, value = torch.chunk(kv, chunks=2, dim=-1)  # B x S x C, B x S x C

    #     query = query * self.scaling  # B x S x C

    #     query = torch.chunk(query, chunks=self.num_heads, dim=-1)  # [B x S x C_split] * {head}
    #     value = torch.chunk(value, chunks=self.num_heads, dim=-1)  # [B x S x C_split] * {head}
    #     key = torch.chunk(key, chunks=self.num_heads, dim=-1)  # [B x S x C_split] * {head}

    #     wt_out = []
    #     for h in range(self.num_heads):
    #         attn_h = torch.matmul(query[h], key[h].transpose(-1, -2))  # B x S x S
    #         attn_h = self.softmax(attn_h)  # B x S x S
    #         attn_h = self.attn_dropout(attn_h)  #  B x S x S
    #         out_h = torch.matmul(attn_h, value[h])  # B x S x C_split
    #         wt_out.append(out_h)

    #     # wt_out = [B x S x C] * {head} 
    #     wt_out = torch.cat(wt_out, dim=-1)  # B x S x C
    #     wt_out = self.out_proj(wt_out)  # B x S x C_out
    #     return wt_out  # B x S x C_out

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Let S_s(source): S_q(query)
        # Let S_t(target): S_k(key) (=S_v(value))
        # For self-attention, S_s = S_t = S
        # Let C: C_split * {head}(=num_heads)
        # x_q: B x S x C
        b_sz, S_len, in_channels = x_q.shape  # B x S_s x C

        if x_kv is None:
            # self-attention
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)  # B x S_s x 3 x {head} x C_split
            qkv = qkv.transpose(1, 3).contiguous()  # B x {head} x 3 x S_s x C_split
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # B x {head} x S_s x C_split, B x {head} x S_t(=S_s) x C_split, B x {head} x S_t(=S_s) x C_split
        # else:
        #     # x_kv: B x S_t x C
        #     T_len = x_kv.shape[1]  

        #     # cross-attention
        #     # [N, S, C]
        #     query = F.linear(
        #         x_q,
        #         weight=self.qkv_proj.weight[: self.embed_dim, ...],
        #         bias=self.qkv_proj.bias[: self.embed_dim]
        #         if self.qkv_proj.bias is not None
        #         else None,
        #     )  # B x S_s x C
        #     # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
        #     query = (
        #         query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
        #         .transpose(1, 2)
        #         .contiguous()
        #     )  # B x {head} x S_s x C_split

        #     # [N, T, C] --> [N, T, 2C]
        #     kv = F.linear(
        #         x_kv,
        #         weight=self.qkv_proj.weight[self.embed_dim :, ...],
        #         bias=self.qkv_proj.bias[self.embed_dim :]
        #         if self.qkv_proj.bias is not None
        #         else None,
        #     )  # B x S_t x 2C
        #     # [N, T, 2C] --> [N, T, 2, h, c]
        #     kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)  # B x S_t x 2 x {head} x C_split
        #     # [N, T, 2, h, c] --> [N, h, 2, T, c]
        #     kv = kv.transpose(1, 3).contiguous()  # B x {head} x 2 x S_t x C_split
        #     key, value = kv[:, :, 0], kv[:, :, 1]  # B x {head} x S_t x C_split, B x {head} x S_t x C_split

        # For self-attention, S_s = S_t
        query = query * self.scaling  # B x {head} x S_s x C_split 
        key = key.transpose(-1, -2)  # B x {head} x C_split x S_t(=S_s)

        attn = torch.matmul(query, key)  # B x {head} x S_s x S_t(=S_s)

        if attn_mask is not None:
            # attn_mask: B x S_s x S_t(=S_s)
            attn_mask = attn_mask.unsqueeze(1)  # B x 1 x S_s x S_t(=S_s)
            attn = attn + attn_mask  # B x {head} x S_s x S_t(=S_s)

        if key_padding_mask is not None:
            # key_padding_mask: B x S_t(=S_s)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),
                float("-inf"),
            )  # B x {head} x S_s x S_t(=S_s)

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())  # B x {head} x S_s x S_t(=S_s)
        attn = attn_as_float.to(attn_dtype)  # B x {head} x S_s x S_t(=S_s)
        attn = self.attn_dropout(attn)  # B x {head} x S_s x S_t(=S_s)

        # weighted sum
        out = torch.matmul(attn, value)    # B x {head} x S_s x C_split

        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)  #  B x S_s x C
        out = self.out_proj(out)  # B x S_s x C_out

        return out  # B x S_s x C_out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward_default(
            x_q=x_q,
            x_kv=x_kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

    # def profile_module(self, input) -> Tuple[Tensor, float, float]:
    #     b_sz, seq_len, in_channels = input.shape
    #     params = macs = 0.0

    #     qkv, p, m = module_profile(module=self.qkv_proj, x=input)
    #     params += p
    #     macs += m * seq_len * b_sz

    #     # number of operations in QK^T
    #     m_qk = (seq_len * seq_len * in_channels) * b_sz
    #     macs += m_qk

    #     # number of operations in computing weighted sum
    #     m_wt = (seq_len * seq_len * in_channels) * b_sz
    #     macs += m_wt

    #     out_p, p, m = module_profile(module=self.out_proj, x=input)
    #     params += p
    #     macs += m * seq_len * b_sz

    #     return input, params, macs

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