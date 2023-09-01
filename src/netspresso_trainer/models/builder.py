import os
from typing import List, Dict, Type, Callable
from pathlib import Path
import logging

import torch
import torch.nn as nn

from .base import TaskModel, ClassificationModel, SegmentationModel, DetectionModel
from .registry import SUPPORTING_TASK_LIST, MODEL_FULL_DICT
from .utils import load_from_checkpoint, load_model_name

logger = logging.getLogger("netspresso_trainer")




def load_full_model(conf_model, model_name, num_classes, model_checkpoint):
    model_fn: Callable[..., nn.Module] = MODEL_FULL_DICT[model_name]
    model: nn.Module = model_fn(conf_model, num_classes)

    model = load_from_checkpoint(model, model_checkpoint)

    return model


def load_backbone_and_head_model(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size, freeze_backbone):
    TASK_MODEL_DICT: Dict[str, Type[TaskModel]] = {
        'classification': ClassificationModel,
        'segmentation': SegmentationModel,
        'detection': DetectionModel
    }

    if not task in TASK_MODEL_DICT:
        raise ValueError(f"No such task(s) named: {task}. This should be included in SUPPORTING_TASK_LIST ({SUPPORTING_TASK_LIST})")

    return TASK_MODEL_DICT[task](conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size, freeze_backbone)


def build_model(conf_model, task, num_classes, model_checkpoint, img_size) -> nn.Module:

    if conf_model.architecture.full is not None:
        model_name = load_model_name(conf_model.architecture.full)
        return load_full_model(conf_model, model_name, num_classes, model_checkpoint)

    backbone_name = load_model_name(conf_model.architecture.backbone)
    head_name = load_model_name(conf_model.architecture.head)
    freeze_backbone = conf_model.freeze_backbone
    return load_backbone_and_head_model(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size, freeze_backbone)
