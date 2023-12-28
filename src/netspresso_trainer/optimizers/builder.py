from typing import Literal

import torch.nn as nn
from omegaconf import DictConfig

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
