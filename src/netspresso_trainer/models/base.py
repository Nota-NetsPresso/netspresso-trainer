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

import os
from abc import abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf

from .utils import BackboneOutput, DetectionModelOutput, ModelOutput


class TaskModel(nn.Module):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone: bool = False) -> None:
        super(TaskModel, self).__init__()
        self.task = conf_model.task
        self.name = conf_model.name
        self.backbone_name = conf_model.architecture.backbone.name
        if neck:
            self.neck_name = conf_model.architecture.neck.name
        self.head_name = conf_model.architecture.head.name

        self.backbone = backbone
        if neck:
            self.neck = neck
        self.head = head

        if freeze_backbone:
            self._freeze_backbone()
            logger.info(f"Freeze! {self.backbone_name} is now freezed. Now only tuning with {self.head_name}.")

    def _freeze_backbone(self):
        for m in self.backbone.parameters():
            m.requires_grad = False

    @property
    def head_list(self):
        return ('head')

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_name(self):
        if hasattr(self, 'neck'):
            return f"{self.__class__.__name__}[task={self.task}, backbone={self.backbone_name}, neck={self.neck_name}, head={self.head_name}]"
        else:
            return f"{self.__class__.__name__}[task={self.task}, backbone={self.backbone_name}, head={self.head_name}]"

    @abstractmethod
    def forward(self, x, label_size=None, targets=None):
        raise NotImplementedError

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


class ClassificationModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['last_feature'])
        return out


class SegmentationModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: ModelOutput = self.head(features['intermediate_features'], targets)
        return out


class DetectionModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: DetectionModelOutput = self.head(features['intermediate_features'], targets)
        return out


class PoseEstimationModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: DetectionModelOutput = self.head(features['intermediate_features'], targets)
        return out


class ONNXModel:
    '''
        ONNX Model wrapper class for inferencing.
    '''
    def __init__(self, model_conf) -> None:
        import onnxruntime as ort
        self.name = model_conf.name + '_onnx'
        self.onnx_path = model_conf.checkpoint.path
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
            }),
            'CPUExecutionProvider',
        ]
        self.inference_session = ort.InferenceSession(model_conf.checkpoint.path, providers=providers)

    def _get_name(self):
        return f"{self.__class__.__name__}[model={self.name}]"

    def __call__(self, x, label_size=None, targets=None):
        device = x.device
        x = x.detach().cpu().numpy()
        out = self.inference_session.run(None, {self.inference_session.get_inputs()[0].name: x})
        out = [torch.tensor(o).to(device) for o in out]

        if len(out) == 1:
            out = out[0]

        return ModelOutput(pred=out)

    def eval(self):
        pass # Do nothing

    def set_provider(self, device):
        if device.type == 'cuda':
            self.inference_session.set_providers(['CUDAExecutionProvider'])
        else:
            self.inference_session.set_providers(['CPUExecutionProvider'])
