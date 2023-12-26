""" Inspired from https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
"""
from typing import Callable, Literal, Optional, Tuple

from omegaconf import DictConfig
import torch.nn as nn

from .registry import OPTIMIZER_DICT


def build_optimizer(
    model_or_params,
    optimizer_conf: DictConfig,
):
    parameters = model_or_params.parameters() if isinstance(model_or_params, nn.Module) else model_or_params

    opt_name: Literal['sgd', 'adam', 'adamw', 'adamax', 'adadelta', 'adagrad', 'rmsprop'] = optimizer_conf.name.lower()
    assert opt_name in OPTIMIZER_DICT

    optimizer = OPTIMIZER_DICT[opt_name](parameters, optimizer_conf)
    return optimizer
