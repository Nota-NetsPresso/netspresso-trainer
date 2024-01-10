import os
from pathlib import Path
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf

from .base import ClassificationModel, DetectionModel, SegmentationModel, TaskModel
from .registry import (
    MODEL_BACKBONE_DICT,
    MODEL_FULL_DICT,
    MODEL_HEAD_DICT,
    MODEL_NECK_DICT,
    SUPPORTING_TASK_LIST,
    TASK_MODEL_DICT,
)
from .utils import load_from_checkpoint


def load_full_model(conf_model, model_name, num_classes, model_checkpoint):
    model_fn: Callable[..., nn.Module] = MODEL_FULL_DICT[model_name]
    model: nn.Module = model_fn(num_classes=num_classes, conf_model_full=conf_model.architecture.full)
    model = load_from_checkpoint(model, model_checkpoint, conf_model.load_checkpoint_head)

    return model


def load_backbone_and_head_model(
        conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size, freeze_backbone):
    if task not in TASK_MODEL_DICT:
        raise ValueError(
            f"No such task(s) named: {task}. This should be included in SUPPORTING_TASK_LIST ({SUPPORTING_TASK_LIST})")

    # Backbone construction
    backbone_fn: Callable[..., nn.Module] = MODEL_BACKBONE_DICT[backbone_name]
    backbone: nn.Module = backbone_fn(task=task, conf_model_backbone=conf_model.architecture.backbone)

    # Neck construction
    intermediate_features_dim = backbone.intermediate_features_dim
    neck = None
    if getattr(conf_model.architecture, 'neck', None):
        neck_name = conf_model.architecture.neck.name
        neck_fn: Callable[..., nn.Module] = MODEL_NECK_DICT[neck_name]
        neck = neck_fn(intermediate_features_dim=backbone.intermediate_features_dim, conf_model_neck=conf_model.architecture.neck)
        intermediate_features_dim = neck.intermediate_features_dim

    # Head construction
    head_module = MODEL_HEAD_DICT[task][head_name]
    if task == 'classification':
        head = head_module(num_classes=num_classes, feature_dim=backbone.feature_dim, conf_model_head=conf_model.architecture.head)
    elif task in ['segmentation', 'detection']:
        img_size = img_size if isinstance(img_size, (int, None)) else tuple(img_size)
        head = head_module(num_classes=num_classes,
                                intermediate_features_dim=intermediate_features_dim,
                                label_size=img_size,
                                conf_model_head=conf_model.architecture.head)

    # Assemble model and load checkpoint
    model = TASK_MODEL_DICT[task](conf_model, backbone, neck, head, freeze_backbone)
    model = load_from_checkpoint(model, model_checkpoint, conf_model.load_checkpoint_head)
    return model


def build_model(conf_model, task, num_classes, model_checkpoint, img_size) -> nn.Module:

    if conf_model.single_task_model:
        model_name = str(conf_model.architecture.full.name).lower()
        return load_full_model(conf_model, model_name, num_classes, model_checkpoint)

    backbone_name = str(conf_model.architecture.backbone.name).lower()
    head_name = str(conf_model.architecture.head.name).lower()
    freeze_backbone = conf_model.freeze_backbone
    return load_backbone_and_head_model(
        conf_model, task, backbone_name, head_name, num_classes, model_checkpoint, img_size, freeze_backbone)
