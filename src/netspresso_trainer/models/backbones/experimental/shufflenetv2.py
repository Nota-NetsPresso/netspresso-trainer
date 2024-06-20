from typing import Dict, Optional, List 
from omegaconf import DictConfig
import torch
import torch.nn as nn 

from ...op.custom import ConvLayer, ShuffleV2Block
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST


__all__ = []
SUPPORTING_TASK = ['detection']

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        # Check task compatibility 
        self.task = task.lower() 
        assert self.task in SUPPORTING_TASK, f"ShuffleNetV2 is not supported on {self.task} task now."
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST
        super().__init__()

        self.stage_repeats = [4, 8, 4]
        model_size = params.model_size 

        if model_size == '0.5x': 
            self.stage_out_channels = [-1, 24, 48, 96, 192]
        elif model_size == '1.0x': 
            self.stage_out_channels = [-1, 24, 116, 232, 464]
        elif model_size == '1.5x': 
            self.stage_out_channels = [-1, 24, 176, 352, 704]
        elif model_size == '2.0x': 
            self.stage_out_channels = [-1 ,24, 244, 488, 976]
        
        stage_names = ["stage2", "stage3", "stage4"]
        self._feature_dim = self.stage_out_channels[-1]
        self._intermediate_features_dim = [self.stage_out_channels[-2], self.stage_out_channels[-1]]
        # building first layer 
        input_dim = self.stage_out_channels[1]
        self.first_conv = ConvLayer(3, input_dim, kernel_size=3, stride=2, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for stage_idx in range(len(self.stage_repeats)):
            num_repeat = self.stage_repeats[stage_idx]
            output_channel = self.stage_out_channels[stage_idx+2]
            stageSeq = []
            for i in range(num_repeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_dim, output_channel, 
                                                hidden_channels=output_channel // 2, kernel_size=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_dim // 2, output_channel, 
                                                hidden_channels=output_channel // 2, kernel_size=3, stride=1))
                input_dim = output_channel
            setattr(self, stage_names[stage_idx], nn.Sequential(*stageSeq))

        self.avgpool = nn.AdaptiveAvgPool2d(1) if not self.use_intermediate_features else None  
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)
        if self.use_intermediate_features: 
            return BackboneOutput(intermediate_features=[C2, C3])

        x = self.avgpool(C3)
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


def shufflenetv2(task, conf_model_backbone) -> ShuffleNetV2: 
    return ShuffleNetV2(task, conf_model_backbone.params, conf_model_backbone.stage_params)

