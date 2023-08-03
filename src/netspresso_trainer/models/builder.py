import os
from typing import List, Dict, Type, Callable
from pathlib import Path

import torch
import torch.nn as nn

from .base import TaskModel, ClassificationModel, SegmentationModel, DetectionModel
from .registry import SUPPORTING_TASK_LIST, MODEL_FULL_DICT
from ..utils.logger import set_logger

logger = set_logger('models', level=os.getenv('LOG_LEVEL', 'INFO'))

def load_full_model(conf_model, model_name, num_classes, model_checkpoint):
    model_fn: Callable[..., nn.Module] = MODEL_FULL_DICT[model_name]
    model: nn.Module = model_fn(conf_model, num_classes)

    if model_checkpoint is not None:
        model_state_dict = torch.load(model_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        
        if len(missing_keys) != 0:
            logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
        if len(unexpected_keys) != 0:
            logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        
    return model

def load_backbone_and_head_model(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size):
    TASK_MODEL_DICT: Dict[str, Type[TaskModel]] = {
        'classification': ClassificationModel,
        'segmentation': SegmentationModel,
        'detection': DetectionModel
    }
    
    if not task in TASK_MODEL_DICT:
        raise ValueError(f"No such task(s) named: {task}. This should be included in SUPPORTING_TASK_LIST ({SUPPORTING_TASK_LIST})")

    return TASK_MODEL_DICT[task](conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size)


def build_model(conf_model, task, num_classes, model_checkpoint, img_size) -> nn.Module:
    
    if conf_model.architecture.full is not None:
        model_name = conf_model.architecture.full
        return load_full_model(conf_model, model_name, num_classes, model_checkpoint)
    
    backbone_name = str(conf_model.architecture.backbone).lower()
    head_name = str(conf_model.architecture.head).lower()
    
    return load_backbone_and_head_model(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size)
