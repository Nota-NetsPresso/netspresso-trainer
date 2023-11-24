"""
Based on the Darknet implementation of Megvii.
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py
"""
from typing import Dict, Optional, List

from omegaconf import DictConfig
import torch
from torch import nn

from ...op.custom import ConvLayer, CSPLayer, Focus, SPPBottleneck
from ...utils import BackboneOutput

__all__ = ['cspdarknet']
SUPPORTING_TASK = ['detection']


class CSPDarknet(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
        #depthwise=False,
    ) -> None:
        super().__init__()
        out_features=("dark3", "dark4", "dark5")
        assert out_features, "please provide output features of Darknet"

        self.task = task.lower()
        self.use_intermediate_features = self.task in ['segmentation', 'detection']

        dep_mul = params.dep_mul
        wid_mul = params.wid_mul
        act_type = params.act_type

        self.out_features = out_features
        Conv = ConvLayer

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act_type=act_type)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(in_channels=base_channels, 
                 out_channels=base_channels * 2, 
                 kernel_size=3, 
                 stride=2, 
                 act_type=act_type),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                #depthwise=depthwise,
                act_type=act_type,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(in_channels=base_channels * 2, 
                 out_channels=base_channels * 4, 
                 kernel_size=3, 
                 stride=2, 
                 act_type=act_type),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                #depthwise=depthwise,
                act_type=act_type,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(in_channels=base_channels * 4,
                 out_channels=base_channels * 8, 
                 kernel_size=3, 
                 stride=2, 
                 act_type=act_type),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                #depthwise=depthwise,
                act_type=act_type,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(in_channels=base_channels * 8, 
                 out_channels=base_channels * 16, 
                 kernel_size=3, 
                 stride=2, 
                 act_type=act_type),
            SPPBottleneck(base_channels * 16, base_channels * 16, act_type=act_type),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                #depthwise=depthwise,
                act_type=act_type,
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        predefined_out_features = {'dark2': base_channels * 2, 'dark3': base_channels * 4, 
                                   'dark4': base_channels * 8, 'dark5': base_channels * 16}
        self._feature_dim = predefined_out_features['dark5']
        self._intermediate_features_dim = [predefined_out_features[out_feature] for out_feature in out_features]

    def forward(self, x):
        outputs_dict = {}
        x = self.stem(x)
        outputs_dict["stem"] = x
        x = self.dark2(x)
        outputs_dict["dark2"] = x
        x = self.dark3(x)
        outputs_dict["dark3"] = x
        x = self.dark4(x)
        outputs_dict["dark4"] = x
        x = self.dark5(x)
        outputs_dict["dark5"] = x

        if self.use_intermediate_features:
            all_hidden_states = [outputs_dict[out_name] for out_name in self.out_features]
            return BackboneOutput(intermediate_features=all_hidden_states)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return BackboneOutput(last_feature=x)
    
    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def cspdarknet(task, conf_model_backbone) -> CSPDarknet:
    return CSPDarknet(task, conf_model_backbone.params, conf_model_backbone.stage_params)
