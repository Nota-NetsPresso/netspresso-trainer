import math
from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn


from ....op.custom import ConvLayer
from ....utils import AnchorBasedDetectionModelOutput
from .detection import AnchorGenerator


class YoloFastestHead(nn.Module):

    num_layers: int

    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()

        anchors = params.anchors
        num_anchors = len(anchors[0]) // 2
        num_layers = len(anchors)

        self.num_layers = num_layers

        out_channels = num_anchors * (5 + num_classes)
        norm_type = params.norm_type
        use_act = False
        kernel_size = 1

        for i in range(num_layers):

            in_channels = intermediate_features_dim[i]

            conv_norm = ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                use_act=use_act,
            )
            conv = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_norm=False,
                use_act=use_act,
            )

            layer = nn.Sequential(conv_norm, conv)

            setattr(self, f"layer_{i+1}", layer)

        def init_bn(M):
            for m in M.modules():

                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_bn)

    def forward(self, inputs: List[torch.Tensor]):

        x1, x2 = inputs
        out1 = self.layer_1(x1)
        out2 = self.layer_2(x2)

        return out1, out2


def yolo_fastest_head(
    num_classes, intermediate_features_dim, conf_model_head, **kwargs
) -> YoloFastestHead:
    return YoloFastestHead(
        num_classes=num_classes,
        intermediate_features_dim=intermediate_features_dim,
        params=conf_model_head.params,
    )
