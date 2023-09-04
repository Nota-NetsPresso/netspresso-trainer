import os
from typing import List, Dict, Type, Callable
from pathlib import Path
import logging

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .base import TaskModel, ClassificationModel, SegmentationModel, DetectionModel
from .registry import SUPPORTING_TASK_LIST, MODEL_FULL_DICT
from .utils import load_from_checkpoint

logger = logging.getLogger("netspresso_trainer")


def load_full_model(conf_model, model_name, num_classes, model_checkpoint):
    model_fn: Callable[..., nn.Module] = MODEL_FULL_DICT[model_name]
    conf_model_full = OmegaConf.to_object(conf_model.architecture.full)
    model: nn.Module = model_fn(num_classes=num_classes, conf_model_full=conf_model_full)
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

    if conf_model.single_task_model:
        model_name = str(conf_model.architecture.full.name).lower()
        return load_full_model(conf_model, model_name, num_classes, model_checkpoint)

    backbone_name = str(conf_model.architecture.backbone.name).lower()
    head_name = str(conf_model.architecture.head.name).lower()
    freeze_backbone = conf_model.freeze_backbone
    return load_backbone_and_head_model(conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size, freeze_backbone)
