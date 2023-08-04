""" Inspired from https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
"""
from typing import Optional, Callable, Tuple, Literal

from .registry import OPTIMIZER_DICT

import torch.nn as nn


def build_optimizer(
    model_or_params,
    opt: str = 'adamw',
    lr: Optional[float] = None,
    wd: float = 0.,
    momentum: float = 0.9,
):

    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_name: Literal['sgd', 'nesterov', 'momentum',
                      'adam', 'adamw', 'adamax',
                      'adadelta', 'adagrad', 'rmsprop'] = opt.lower()
    assert opt_name in OPTIMIZER_DICT
    
    conf_optim = {'weight_decay': wd, 'lr': lr}

    if opt_name in ['sgd', 'nesterov', 'momentum', 'rmsprop']:
        conf_optim.update({'momentum': momentum})
    if opt_name in ['rmsprop']:
        conf_optim.update({'alpha': 0.9})
    if opt_name in ['sgd', 'nesterov']:
        conf_optim.update({'nesterov': True})
    if opt_name in ['momentum']:
        conf_optim.update({'nesterov': False})
        
    optimizer = OPTIMIZER_DICT[opt_name](parameters, **conf_optim)

    return optimizer
