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
from pathlib import Path
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import ClassificationModel, DetectionModel, ONNXModel, SegmentationModel, TaskModel
from .registry import (
    MODEL_BACKBONE_DICT,
    MODEL_FULL_DICT,
    MODEL_HEAD_DICT,
    MODEL_NECK_DICT,
    SUPPORTING_TASK_LIST,
    TASK_MODEL_DICT,
)
from .utils import get_model_format, load_from_checkpoint


def load_full_model(conf_model, model_name, num_classes, model_checkpoint, use_pretrained):
    model_fn: Callable[..., nn.Module] = MODEL_FULL_DICT[model_name]
    conf_model.architecture.full.nick_name = conf_model.name # @illian01: model.name should be same with conf_model.name
    model: nn.Module = model_fn(num_classes=num_classes, conf_model_full=conf_model.architecture.full)
    if use_pretrained:
        model = load_from_checkpoint(
            model, model_checkpoint,
            load_checkpoint_head=conf_model.checkpoint.load_head,
        )
        # TODO: Move to model property
        model.save_dtype = next(model.parameters()).dtype # If loaded model is float16, save it as float16
        model = model.float() # Train with float32

    return model


def load_backbone_and_head_model(
        conf_model, task, backbone_name, head_name, num_classes,
        model_checkpoint, use_pretrained, freeze_backbone,
    ):
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
    elif task in ['segmentation', 'detection', 'pose_estimation']:
        head = head_module(num_classes=num_classes,
                                intermediate_features_dim=intermediate_features_dim,
                                conf_model_head=conf_model.architecture.head)

    # Assemble model and load checkpoint
    model = TASK_MODEL_DICT[task](conf_model, backbone, neck, head, freeze_backbone)
    if use_pretrained:
        model = load_from_checkpoint(
            model, model_checkpoint,
            load_checkpoint_head=conf_model.checkpoint.load_head,
        )
        # TODO: Move to model property
        model.save_dtype = next(model.parameters()).dtype # If loaded model is float16, save it as float16
        model = model.float() # Train with float32
    return model


def build_model(model_conf, num_classes, devices, distributed) -> nn.Module:

    task = model_conf.task
    model_checkpoint = model_conf.checkpoint.path
    use_pretrained = model_conf.checkpoint.use_pretrained

    model_format = get_model_format(model_conf)

    if model_format == 'torch':
        if model_conf.single_task_model:
            model_name = str(model_conf.architecture.full.name).lower()
            model = load_full_model(
                model_conf, model_name, num_classes,
                model_checkpoint=model_checkpoint,
                use_pretrained=use_pretrained
            )
        else:

            backbone_name = str(model_conf.architecture.backbone.name).lower()
            head_name = str(model_conf.architecture.head.name).lower()
            freeze_backbone = model_conf.freeze_backbone
            model = load_backbone_and_head_model(
                model_conf, task, backbone_name, head_name, num_classes,
                model_checkpoint=model_checkpoint,
                use_pretrained=use_pretrained,
                freeze_backbone=freeze_backbone,
            )

        model = model.to(device=devices)
        if distributed:
            model = DDP(model, device_ids=[devices], find_unused_parameters=True)  # TODO: find_unused_parameters should be false (for now, PIDNet has problem)

        return model

    elif model_format == 'torch.fx':
        assert Path(model_conf.checkpoint.path).exists()
        model = torch.load(model_conf.checkpoint.path)

    elif model_format == 'onnx':
        assert Path(model_conf.checkpoint.path).exists()
        model = ONNXModel(model_conf)
        model.set_provider(devices)

    return model
