import os
from typing import Type, Callable
from pathlib import Path
from abc import abstractmethod

import torch
import torch.nn as nn

from models.registry import (
    SUPPORTING_MODEL_LIST, MODEL_PRETRAINED_DICT, MODEL_BACKBONE_DICT, MODEL_HEAD_DICT
)
from utils.logger import set_logger
logger = set_logger('models', level=os.getenv('LOG_LEVEL', 'INFO'))

UPDATE_PREFIX = "updated_"


def load_pretrained_checkpoint(model_name: str):
    assert model_name in SUPPORTING_MODEL_LIST, f"The model_name ({model_name}) should be one of the followings: {SUPPORTING_MODEL_LIST}"

    if not model_name in MODEL_PRETRAINED_DICT:
        raise AssertionError(f"No pretrained checkpoint found! Model name: ({model_name})")

    state_dict_path: Path = MODEL_PRETRAINED_DICT[model_name]
    state_dict = torch.load(state_dict_path)
    return state_dict


class TaskModel(nn.Module):
    def __init__(self, args, task, backbone_name, head_name, num_classes) -> None:
        super(TaskModel, self).__init__()
        self.task = task
        self.backbone_name = backbone_name
        self.head_name = head_name
        
        backbone_fn: Callable[..., nn.Module] = MODEL_BACKBONE_DICT[backbone_name]
        self.backbone = backbone_fn(task=self.task)
        
        try:
            model_state_dict = load_pretrained_checkpoint(backbone_name)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_state_dict, strict=False)
            logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
            logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        except AssertionError as e:
            logger.warning(str(e))
        # self._freeze_backbone()
        
        head_module = MODEL_HEAD_DICT[self.task][head_name]
        self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)
        
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
    def __init__(self, args, task, backbone_name, head_name, num_classes) -> None:
        super().__init__(args, task, backbone_name, head_name, num_classes)
    
    def forward(self, x, label_size=None, targets=None):
        features = self.backbone(x)
        out = self.head(features['last_feature'])
        return out


class SegmentationModel(TaskModel):
    def __init__(self, args, task, backbone_name, head_name, num_classes) -> None:
        super().__init__(args, task, backbone_name, head_name, num_classes)
    
    def forward(self, x, label_size=None, targets=None):
        features = self.backbone(x)
        out = self.head(features['intermediate_features'], label_size=label_size)
        return out


class DetectionModel(TaskModel):
    def __init__(self, args, task, backbone_name, head_name, num_classes) -> None:
        super().__init__(args, task, backbone_name, head_name, num_classes)
    
    def forward(self, x, label_size=None, targets=None):
        features = self.backbone(x)
        out = self.head(features['intermediate_features'], targets=targets)
        return out
    