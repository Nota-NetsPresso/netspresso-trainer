import os
from typing import List, Dict, Type, Callable
from pathlib import Path

import torch
import torch.nn as nn

from .base import TaskModel, ClassificationModel, SegmentationModel, DetectionModel
from .registry import SUPPORTING_TASK_LIST, MODEL_FULL_DICT
from ..utils.logger import set_logger

logger = set_logger('models', level=os.getenv('LOG_LEVEL', 'INFO'))

def load_full_model(args, model_name, num_classes, model_checkpoint):
    model_fn: Callable[..., nn.Module] = MODEL_FULL_DICT[model_name]
    model: nn.Module = model_fn(args, num_classes)

    if model_checkpoint is not None:
        model_state_dict = torch.load(model_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        
        if len(missing_keys) != 0:
            logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
        if len(unexpected_keys) != 0:
            logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        
    return model

def load_backbone_and_head_model(args, task, backbone_name, head_name, num_classes, model_checkpoint):
    TASK_MODEL_DICT: Dict[str, Type[TaskModel]] = {
        'classification': ClassificationModel,
        'segmentation': SegmentationModel,
        'detection': DetectionModel
    }
    
    if not task in TASK_MODEL_DICT:
        raise ValueError(f"No such task(s) named: {task}. This should be included in SUPPORTING_TASK_LIST ({SUPPORTING_TASK_LIST})")

    return TASK_MODEL_DICT[task](args, task, backbone_name, head_name, num_classes, model_checkpoint)


def build_model(args, num_classes, model_checkpoint) -> nn.Module:
    
    if args.model.architecture.full is not None:
        model_name = args.model.architecture.full
        return load_full_model(args, model_name, num_classes, model_checkpoint)
    
    task = str(args.model.task).lower()
    assert task in SUPPORTING_TASK_LIST
    
    backbone_name = str(args.model.architecture.backbone).lower()
    head_name = str(args.model.architecture.head).lower()
    
    return load_backbone_and_head_model(args, task, backbone_name, head_name, num_classes, model_checkpoint)
