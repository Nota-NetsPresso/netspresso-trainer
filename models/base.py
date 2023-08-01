import os
from typing import Callable, Union
from pathlib import Path
from abc import abstractmethod

import torch
import torch.nn as nn

from models.registry import MODEL_BACKBONE_DICT, MODEL_HEAD_DICT
from models.utils import BackboneOutput, ModelOutput, PIDNetModelOutput

from utils.logger import set_logger
logger = set_logger('models', level=os.getenv('LOG_LEVEL', 'INFO'))


class TaskModel(nn.Module):
    def __init__(self, args, task, backbone_name, head_name, num_classes, model_checkpoint) -> None:
        super(TaskModel, self).__init__()
        self.task = task
        self.backbone_name = backbone_name
        self.head_name = head_name
        
        backbone_fn: Callable[..., nn.Module] = MODEL_BACKBONE_DICT[backbone_name]
        self.backbone = backbone_fn(task=self.task)
        
        if model_checkpoint is not None:
            model_state_dict = torch.load(model_checkpoint)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_state_dict, strict=False)
            
            if len(missing_keys) != 0:
                logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
            if len(unexpected_keys) != 0:
                logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        
        head_module = MODEL_HEAD_DICT[self.task][head_name]
        label_size = args.training.img_size if task in ['segmentation', 'detection'] else None
        self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes, label_size=label_size)
        
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
    def __init__(self, args, task, backbone_name, head_name, num_classes, model_checkpoint) -> None:
        super().__init__(args, task, backbone_name, head_name, num_classes, model_checkpoint)
    
    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['last_feature'])
        return out


class SegmentationModel(TaskModel):
    def __init__(self, args, task, backbone_name, head_name, num_classes, model_checkpoint) -> None:
        super().__init__(args, task, backbone_name, head_name, num_classes, model_checkpoint)
    
    def forward(self, x, label_size=None, targets=None):
        features: BackboneOutput = self.backbone(x)
        out: ModelOutput = self.head(features['intermediate_features'])
        return out


class DetectionModel(TaskModel):
    def __init__(self, args, task, backbone_name, head_name, num_classes, model_checkpoint) -> None:
        super().__init__(args, task, backbone_name, head_name, num_classes, model_checkpoint)
    
    def forward(self, x, label_size=None, targets=None):
        features = self.backbone(x)
        out = self.head(features['intermediate_features'], targets=targets)
        return out
    