"""
Based on the Darknet implementation of Megvii.
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/darknet.py
"""

from typing import Dict, Optional, List, Type

from omegaconf import DictConfig
import torch
from torch import nn

from ...op.custom import (
    ConvLayer,
    CSPLayer,
    Focus,
    SPPBottleneck,
    Bottleneck,
    BasicBlock,
    DarknetBlock,
)
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ["cspdarknet"]
SUPPORTING_TASK = ["classification", "segmentation", "detection", "pose_estimation"]


BLOCK_FROM_LITERAL: Dict[str, Type[nn.Module]] = {
    "basicblock": BasicBlock,
    "bottleneck": Bottleneck,
    "darknetblock": DarknetBlock,
}

DARKNET_SUPPORTED_BLOCKS = ["bottleneck"]


class CSPDarknet(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
        #depthwise=False,
    ) -> None:
        # Check task compatibility
        self.task = task.lower()
        assert self.task in SUPPORTING_TASK, f'CSPDarknet is not supported on {self.task} task now.'
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST

        super().__init__()

        out_features=("dark3", "dark4", "dark5")
        assert out_features, "please provide output features of Darknet"

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

        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_bn)

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

class Darknet(nn.Module):
    """
    Consists of a stem layer and multiple stage layers.
    Stage layers are named as stage_{i} starting from stage_1
    """

    num_layers: int

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        self.task = task.lower()
        assert (
            self.task in SUPPORTING_TASK
        ), f"Darknet is not supported on {self.task} task now."
        assert stage_params, "please provide stage params of Darknet"
        assert len(stage_params) >= 2
        assert (
            params.block_type.lower() in DARKNET_SUPPORTED_BLOCKS
        ), "Block type not supported"
        self.use_intermediate_features = (
            self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST
        )

        self.num_layers = len(stage_params)

        super().__init__()

        # TODO: Check if inplace activation should be used
        act_type = params.act_type
        norm_type = params.norm_type
        block_type = params.block_type

        Block = BLOCK_FROM_LITERAL[block_type.lower()]
        predefined_out_features = dict()

        # build the stem layer
        stem_stage = stage_params[0]

        stem_act = None
        stem_norm = None
        if stem_stage.use_act:
            stem_act = act_type
        if stem_stage.use_norm:
            stem_norm = norm_type

        self.stem = ConvLayer(
            stem_stage.in_channels,
            stem_stage.out_channels,
            stem_stage.kernel_sizes,
            stem_stage.stride,
            act_type=stem_act,
            norm_type=stem_norm,
        )

        # build rest of the layers
        for i, stage_param in enumerate(stage_params[1:]):

            layers = []
            num_layers = len(stage_param.in_channels)
            hidden_expansion = stage_param.hidden_expansion

            for j in range(num_layers - 1):
                in_ch = stage_param.in_channels[j]
                out_ch = stage_param.out_channels[j]
                kernel_size = stage_param.kernel_sizes[j]
                stride = stage_param.stride[j]
                use_act = stage_param.use_act[j]
                use_group = stage_param.use_group[j]
                num_block = stage_param.num_blocks[j]

                for _ in range(num_block):
                    conv_layer = ConvLayer(
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride,
                        use_act=use_act,
                        act_type=act_type if use_act else None,
                        groups=in_ch if use_group else 1,
                    )
                    layers.append(conv_layer)

            for _ in range(stage_param.num_blocks[-1]):
                in_ch = stage_param.in_channels[-1]
                out_ch = stage_param.out_channels[-1]
                use_group = stage_param.use_group[-1]
                darknet_block = Block(
                    in_ch,
                    out_ch,
                    act_type=act_type,
                    no_out_act=True,
                    expansion=1,
                    groups=in_ch,
                    base_width=64 * hidden_expansion / out_ch,
                )

                layers.append(darknet_block)
            setattr(self, f"stage_{i+1}", nn.Sequential(*layers))
            predefined_out_features[f"stage_{i+1}"] = stage_param.out_channels[-1]

        self._feature_dim = predefined_out_features[f"stage_{self.num_layers-1}"]

        intermediate_out_features = []
        for i in range(params.num_feat_layers - 1):
            stage_num = self.num_layers - (i + 2)
            intermediate_out_features.append(f"stage_{stage_num}")

        self._intermediate_features_dim = [
            predefined_out_features[out_feature]
            for out_feature in intermediate_out_features
        ]

        self.out_features = ("stage_5", "stage_6")

        # Initialize
        def init_bn(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.apply(init_bn)

    def forward(self, x):
        outputs_dict = {}
        x = self.stem(x)
        outputs_dict["stem"] = x

        for i in range(1, self.num_layers):
            x = getattr(self, f"stage_{i}")(x)
            outputs_dict[f"stage_{i}"] = x

        if self.use_intermediate_features:
            all_hidden_states = [
                outputs_dict[out_name] for out_name in self.out_features
            ]
            return BackboneOutput(intermediate_features=all_hidden_states)

        # TODO: Check if classification head is needed
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


def darknet(task, conf_model_backbone) -> Darknet:
    return Darknet(task, conf_model_backbone.params, conf_model_backbone.stage_params)
