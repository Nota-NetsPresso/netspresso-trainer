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

"""
Based on the Torchvision implementation of MobileNetV3.
https://pytorch.org/vision/main/_modules/torchvision/models/mobilenetv3.html
"""
from typing import List, Dict, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from ...op.custom import ConvLayer, InvertedResidual
from ...utils import BackboneOutput
from ..registry import USE_INTERMEDIATE_FEATURES_TASK_LIST

__all__ = ['mobilenetv4']

SUPPORTING_TASK = ['classification', 'segmentation', 'detection', 'pose_estimation']


class MobileNetV4(nn.Module):

    def __init__(
        self,
        task: str,
        params: Optional[DictConfig] = None,
        stage_params: Optional[List] = None,
    ) -> None:
        # Check task compatibility
        self.task = task.lower()
        assert self.task in SUPPORTING_TASK, f'MobileNetV4 is not supported on {self.task} task now.'
        self.use_intermediate_features = self.task in USE_INTERMEDIATE_FEATURES_TASK_LIST

        super(MobileNetV4, self).__init__()

        # Dummy
        self._feature_dim = 1024
        self._intermediate_features_dim = [256, 512, 1024]

    def forward(self, x: Tensor):
        return BackboneOutput(last_feature=x)

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim

    def task_support(self, task):
        return task.lower() in SUPPORTING_TASK


def mobilenetv4(task, conf_model_backbone) -> MobileNetV4:
    return MobileNetV4(task, conf_model_backbone.params, conf_model_backbone.stage_params)
