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
        out: ModelOutput = self.head(features['intermediate_features'])
        return out


class DetectionModel(TaskModel):
    def __init__(self, conf_model, backbone, neck, head, freeze_backbone=False) -> None:
        super().__init__(conf_model, backbone, neck, head, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        if hasattr(self, 'neck'):
            features: BackboneOutput = self.neck(features['intermediate_features'])
        out: DetectionModelOutput = self.head(features['intermediate_features'])
        return out
