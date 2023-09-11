import logging
import os
from abc import abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .registry import MODEL_BACKBONE_DICT, MODEL_HEAD_DICT
from .utils import BackboneOutput, DetectionModelOutput, ModelOutput, load_from_checkpoint

logger = logging.getLogger("netspresso_trainer")


class TaskModel(nn.Module):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint,
                 img_size: Optional[Union[int, Tuple]] = None, freeze_backbone: bool = False) -> None:
        super(TaskModel, self).__init__()
        self.task = task
        self.backbone_name = backbone_name
        self.head_name = head_name

        backbone_fn: Callable[..., nn.Module] = MODEL_BACKBONE_DICT[backbone_name]
        conf_model_backbone = OmegaConf.to_object(conf_model.architecture.backbone)
        self.backbone: nn.Module = backbone_fn(task=self.task, conf_model_backbone=conf_model_backbone)

        self.backbone = load_from_checkpoint(self.backbone, model_checkpoint)

        head_module = MODEL_HEAD_DICT[self.task][head_name]
        if task == 'classification':
            self.head = head_module(num_classes=num_classes, feature_dim=self.backbone.feature_dim)
        elif task in ['segmentation', 'detection']:
            img_size = img_size if isinstance(img_size, (int, None)) else tuple(img_size)
            self.head = head_module(num_classes=num_classes,
                                    intermediate_features_dim=self.backbone.intermediate_features_dim,
                                    label_size=img_size)

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
        return f"{self.__class__.__name__}[task={self.task}, backbone={self.backbone_name}, head={self.head_name}]"

    @abstractmethod
    def forward(self, x, label_size=None, targets=None):
        raise NotImplementedError


class ClassificationModel(TaskModel):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint,
                 label_size=None, freeze_backbone=False) -> None:
        super().__init__(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['last_feature'])
        return out


class SegmentationModel(TaskModel):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint,
                 label_size, freeze_backbone=False) -> None:
        super().__init__(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['intermediate_features'])
        return out


class DetectionModel(TaskModel):
    def __init__(self, conf_model, task, backbone_name, head_name, num_classes, model_checkpoint,
                 label_size, freeze_backbone=False) -> None:
        super().__init__(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, label_size, freeze_backbone)

    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: DetectionModelOutput = self.head(features['intermediate_features'], targets=targets)
        return out
