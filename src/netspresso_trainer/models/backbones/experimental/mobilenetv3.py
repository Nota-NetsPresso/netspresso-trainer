from typing import Dict, List, Literal, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['mobilenetv3_small']

SUPPORTING_TASK = ['classification', 'segmentation']


class MobileNetV3(nn.Module):

    def __init__(
        self,
        task: str,
        **kwargs
    ) -> None:
        super(MobileNetV3, self).__init__()
        # TODO
        pass

    def forward(self, x: Tensor):
        # TODO
        return None

    @property
    def feature_dim(self):
        return None

    @property
    def intermediate_features_dim(self):
        return None

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def mobilenetv3_small(task, conf_model_backbone) -> MobileNetV3:
    return MobileNetV3(task, **conf_model_backbone)
