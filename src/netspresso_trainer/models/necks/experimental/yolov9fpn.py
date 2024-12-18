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

from typing import List

from omegaconf import DictConfig
import torch
import torch.nn as nn

from ...op.custom import SPPELAN, ELAN, AConv, ADown
from ...utils import BackboneOutput


class YOLOv9FPN(nn.Module):
    """
        YOLOv9 Path Aggregation Feature Pyramid Network (FPN) module.
        Based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py.
    """
    def __init__(
        self,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()
        self.in_channels = intermediate_features_dim
        repeat_num = params.repeat_num
        act_type = params.act_type
        bu_type = params.bu_type # bottom-up type
        assert bu_type.lower() in ['aconv', 'adown']
        bu_block = AConv if bu_type == 'aconv' else ADown
        spp_channels = params.spp_channels
        n4_channels = params.n4_channels
        p3_channels = params.p3_channels
        p4_channels = params.p4_channels
        p5_channels = params.p5_channels
        p3_to_p4_channels = params.p3_to_p4_channels
        p4_to_p5_channels = params.p4_to_p5_channels
        
        # self.use_aux_loss = params.use_aux_loss
        
        # Top-down pathway (upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.spp_block = SPPELAN(
            in_channels=int(self.in_channels[2]),
            out_channels=spp_channels,
            act_type=act_type,
        )
        # if self.use_aux_loss:
        #     self.aux_spp_block = SPPELAN(
        #         in_channels=int(self.in_channels[2]),
        #         out_channels=spp_channels,
        #         act_type=act_type
        #     )
        # else:
        #     self.aux_spp_block = None
        
        # Top-down fusion blocks
        self.td_fusion_block_1 = ELAN(
            in_channels=int(self.in_channels[1] + spp_channels),
            out_channels=n4_channels,
            part_channels=n4_channels,
            n=repeat_num,
            layer_type="repncsp",
            act_type=act_type,
            use_identity=False
        )

        # if self.use_aux_loss:
        #     self.aux_td_fusion_block_1 = ELAN(
        #         in_channels=int(self.in_channels[1] + spp_channels),
        #         out_channels=n4_channels,
        #         part_channels=n4_channels,
        #         n=repeat_num,
        #         layer_type="repncsp",
        #         act_type=act_type,
        #         use_identity=False
        #     )
        # else:
        #     self.aux_td_fusion_block_1 = None
        
        self.td_fusion_block_2 = ELAN(
            in_channels=int(self.in_channels[0] + n4_channels),
            out_channels=p3_channels,
            part_channels=p3_channels,
            n=repeat_num,
            layer_type="repncsp",
            act_type=act_type,
            use_identity=False
        )

        # if self.use_aux_loss:
        #     self.aux_td_fusion_block_2 = ELAN(
        #         in_channels=int(self.in_channels[0] + n4_channels),
        #         out_channels=p3_channels,
        #         part_channels=p3_channels,
        #         n=repeat_num,
        #         layer_type="repncsp",
        #         act_type=act_type,
        #         use_identity=False
        #     )
        # else:
        #     self.aux_td_fusion_block_2 = None
        
        # Bottom-up pathway (downsampling)
        self.bu_conv_p3_to_p4 = bu_block(
            in_channels=p3_channels,
            out_channels=p3_to_p4_channels,
            act_type=act_type
        )
        
        self.bu_fusion_block_1 = ELAN(
            in_channels=int(p3_to_p4_channels + n4_channels),
            out_channels=p4_channels,
            part_channels=p4_channels,
            n=repeat_num,
            layer_type="repncsp",
            act_type=act_type,
            use_identity=False
        )
        
        self.bu_conv_p4_to_p5 = bu_block(
            in_channels=p4_channels, 
            out_channels=p4_to_p5_channels,
            act_type=act_type
        )
        
        self.bu_fusion_block_2 = ELAN(
            in_channels=int(p4_to_p5_channels + spp_channels),
            out_channels=p5_channels,
            part_channels=p5_channels,
            n=repeat_num,
            layer_type="repncsp",
            act_type=act_type,
            use_identity=False
        )
        
        self._intermediate_features_dim = (p3_channels, p4_channels, p5_channels)
        
        # Initialize BatchNorm layers
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)

    def forward(self, inputs):
        """
        Args:
            inputs: List of backbone features [P3, P4, P5]
        Returns:
            Tuple[Tensor]: FPN features at different scales
        """
        [feat_p3, feat_p4, feat_p5] = inputs
        
        # Top-down pathway
        spp_feat = self.spp_block(feat_p5)  # P5 processing
        td_p4 = self.upsample(spp_feat)  # P5 -> P4
        td_p4_concat = torch.cat([td_p4, feat_p4], 1)
        td_p4_processed = self.td_fusion_block_1(td_p4_concat) # N4
        
        td_p3 = self.upsample(td_p4_processed)  # P4 -> P3
        td_p3_concat = torch.cat([td_p3, feat_p3], 1)
        p3_out = self.td_fusion_block_2(td_p3_concat)
        
        # Bottom-up pathway
        bu_p4 = self.bu_conv_p3_to_p4(p3_out)  # P3 -> P4
        bu_p4_concat = torch.cat([bu_p4, td_p4_processed], 1)
        p4_out = self.bu_fusion_block_1(bu_p4_concat)
        
        bu_p5 = self.bu_conv_p4_to_p5(p4_out)  # P4 -> P5
        bu_p5_concat = torch.cat([bu_p5, spp_feat], 1)
        p5_out = self.bu_fusion_block_2(bu_p5_concat)
        
        outputs = (p3_out, p4_out, p5_out)

        # if self.training and self.use_aux_loss:
        #     spp_a5 = self.aux_spp_block(feat_p5) # A5
        #     aux_td_p4 = self.upsample(spp_a5)
        #     aux_td_p4_concat = torch.cat([aux_td_p4, feat_p4], 1)
        #     td_a4 = self.aux_td_fusion_block_1(aux_td_p4_concat) # A4
        #     aux_td_p3 = self.upsample(td_a4)
        #     aux_td_p3_concat = torch.cat([aux_td_p3, feat_p3], 1)
        #     td_a3 = self.aux_td_fusion_block_2(aux_td_p3_concat) # A3
        #     aux_outputs = (td_a3, td_a4, spp_a5)
        #     outputs = {"outputs": outputs, "aux_outputs": aux_outputs}
        # else:
        #     aux_outputs = None
        return BackboneOutput(intermediate_features=outputs)
    
    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

def yolov9fpn(intermediate_features_dim, conf_model_neck, **kwargs):
    return YOLOv9FPN(intermediate_features_dim=intermediate_features_dim, params=conf_model_neck.params)