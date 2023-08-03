from typing import Any, Dict, Union
from itertools import chain

import torch
import torch.nn as nn

from .registry import LOSS_DICT
from ..utils.record import AverageMeter

MODE = ['train', 'valid', 'test']
IGNORE_INDEX_NONE_AS = -100  # following PyTorch preference

class LossFactory:
    def __init__(self, conf_model, **kwargs) -> None:
        self.loss_func_dict = dict()
        self.loss_weight_dict = dict()

        self.conf_model = conf_model
        self._build_losses()

        self.loss_val_per_epoch = {
            mode: {
                loss_key: AverageMeter(loss_key, ':.4e') for loss_key in chain(self.loss_func_dict.keys(), ['total'])
            }
            for mode in MODE
        }

        self.loss_val_per_step = {
            mode: {
                loss_key: 0. for loss_key in chain(self.loss_func_dict.keys(), ['total'])
            }
            for mode in MODE
        }

        self._clear()

    def _build_losses(self):

        for loss_element in self.conf_model.losses:
            criterion = loss_element.criterion
            loss_config = {k: v for k, v in loss_element.items() if k not in ['criterion', 'weight']}
            loss = LOSS_DICT[criterion](**loss_config)
            
            self.loss_func_dict.update({criterion: loss})
            loss_weight = loss_element.weight if loss_element.weight is not None else 1.0
            self.loss_weight_dict.update({criterion: loss_weight})

    def _clear(self):
        self.total_loss_for_backward = 0  

    def backward(self):
        self.total_loss_for_backward.requires_grad_(True)
        self.total_loss_for_backward.mean().backward()
        
    def _assert_argument(self, kwargs):
        if 'boundary_loss' in self.loss_func_dict:
            assert 'bd_gt' in kwargs, "BoundaryLoss failed!"

    def __call__(self, out: Dict, target: Union[torch.Tensor, Dict[str, torch.Tensor]], mode='train', *args: Any, **kwargs: Any) -> None:
        _mode = mode.lower()
        assert _mode in MODE, f"{_mode} is not defined at our mode list ({MODE})"
        
        bd_gt = kwargs['bd_gt'] if 'bd_gt' in kwargs else None
        self._assert_argument(kwargs)
        self._clear()
        
        for loss_key, loss_func in self.loss_func_dict.items():
            if loss_key == 'boundary_loss':
                loss_val = loss_func(out, bd_gt)
            else:
                loss_val = loss_func(out, target)
            self.loss_val_per_step[_mode][loss_key] = loss_val.item()
            self.loss_val_per_epoch[_mode][loss_key].update(loss_val.item())
            
            self.total_loss_for_backward += loss_val * self.loss_weight_dict[loss_key]

        self.loss_val_per_step[_mode]['total'] = self.total_loss_for_backward.item()
        self.loss_val_per_epoch[_mode]['total'].update(self.total_loss_for_backward.item())

    def result(self, mode='train'):
        _mode = mode.lower()
        return self.loss_val_per_epoch[_mode]


def build_losses(conf_model, **kwargs):
    loss_handler = LossFactory(conf_model, **kwargs)
    return loss_handler
