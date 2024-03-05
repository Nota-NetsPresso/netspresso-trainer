from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ...op.custom import ConvLayer, CSPLayer
from ...utils import BackboneOutput


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()
        
        self.in_channels = intermediate_features_dim
        Conv = ConvLayer

        depth = params.dep_mul
        act_type = params.act_type

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = ConvLayer(
            in_channels=int(self.in_channels[2]), 
            out_channels=int(self.in_channels[1]), 
            kernel_size=1, 
            stride=1,
            act_type=act_type
        )
        self.C3_p4 = CSPLayer(
            in_channels=int(2 * self.in_channels[1]),
            out_channels=int(self.in_channels[1]),
            n=round(3 * depth),
            shortcut=False,
            act_type=act_type,
        )  # cat

        self.reduce_conv1 = ConvLayer(
            in_channels=int(self.in_channels[1]), 
            out_channels=int(self.in_channels[0]), 
            kernel_size=1, 
            stride=1, 
            act_type=act_type
        )
        self.C3_p3 = CSPLayer(
            in_channels=int(2 * self.in_channels[0]),
            out_channels=int(self.in_channels[0]),
            n=round(3 * depth),
            shortcut=False,
            act_type=act_type,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            in_channels=int(self.in_channels[0]), 
            out_channels=int(self.in_channels[0]), 
            kernel_size=3, 
            stride=2, 
            act_type=act_type
        )
        self.C3_n3 = CSPLayer(
            in_channels=int(2 * self.in_channels[0]),
            out_channels=int(self.in_channels[1]),
            n=round(3 * depth),
            shortcut=False,
            act_type=act_type,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            in_channels=int(self.in_channels[1]), 
            out_channels=int(self.in_channels[1]), 
            kernel_size=3, 
            stride=2, 
            act_type=act_type
        )
        self.C3_n4 = CSPLayer(
            in_channels=int(2 * self.in_channels[1]),
            out_channels=int(self.in_channels[2]),
            n=round(3 * depth),
            shortcut=False,
            act_type=act_type,
        )

        self._intermediate_features_dim = self.in_channels

    def forward(self, inputs):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        [x2, x1, x0] = inputs

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return BackboneOutput(intermediate_features=outputs)
    
    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

def yolopafpn(intermediate_features_dim, conf_model_neck, **kwargs):
    return YOLOPAFPN(intermediate_features_dim=intermediate_features_dim, params=conf_model_neck.params)
