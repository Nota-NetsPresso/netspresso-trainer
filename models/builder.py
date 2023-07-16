import importlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import models.full as full
import models.backbones as backbones
import models.heads as heads

from models.registry import SUPPORTING_MODEL_LIST, MODEL_PRETRAINED_DICT
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


class AssembleModel(nn.Module):
    def __init__(self, args, num_classes) -> None:
        super(AssembleModel, self).__init__()
        self.task = args.model.task
        backbone_name = args.model.architecture.backbone
        head = args.model.architecture.head

        self.backbone: nn.Module = eval(f"backbones.{backbone_name}")(task=self.task)
        try:
            model_state_dict = load_pretrained_checkpoint(backbone_name)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_state_dict, strict=False)
            logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
            logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        except AssertionError as e:
            logger.warning(str(e))
        # self._freeze_backbone()

        if self.task == 'classification':
            head_module: nn.Module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)
        if self.task == 'segmentation':
            head_module: nn.Module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)
        if self.task == 'detection':
            head_module: nn.Module = eval(f"heads.{self.task}.{head}")
            self.head = head_module(feature_dim=self.backbone.last_channels, num_classes=num_classes)

    def _freeze_backbone(self):
        for m in self.backbone.parameters():
            m.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, label_size=None, targets=None):
        features = self.backbone(x)
        if self.task == 'classification':
            out = self.head(features['last_feature'])
        elif self.task == 'segmentation':
            out = self.head(features['intermediate_features'], label_size=label_size)
        elif self.task == 'detection':
            out = self.head(features['intermediate_features'], targets=targets)

        return out


def build_model(args, num_classes):
    if args.model.architecture.full is not None:
        model_name = args.model.architecture.full
        model: nn.Module = eval(f"full.{model_name}")(args, num_classes)

        model_state_dict = load_pretrained_checkpoint(model_name)
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        logger.warning(f"Missing key(s) in state_dict: {missing_keys}")
        logger.warning(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        return model

    model = AssembleModel(args, num_classes)
    return model
