from typing import Any
from itertools import chain

import torch
import torch.nn as nn

from losses.classification.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from utils.common import AverageMeter

MODE = ['train', 'valid', 'test']

class LossFactory:
    def __init__(self, args) -> None:
        self.loss_func_dict = dict()
        self._build_losses(args)
        self.loss_val_dict = {
            _mode: {
            loss_key: AverageMeter(loss_key, ':.4e') for loss_key in chain(self.loss_func_dict.keys(), ['total'])
            }
            for _mode in MODE
        }
        self._clear()
        
    def _build_losses(self, args):
        
        # TODO: multiple loss functions
        criterion = args.train.losses.criterion
        
        if criterion == 'cross_entropy':
            loss = nn.CrossEntropyLoss()
        elif criterion == 'soft_target_cross_entropy':
            loss = SoftTargetCrossEntropy()
        elif criterion == 'label_smoothing_cross_entropy':
            loss = LabelSmoothingCrossEntropy(smoothing=args.train.losses.smoothing)  # TODO: condition
        else:
            raise AssertionError(f"Unknown criterion! ({criterion})")

        self.loss_func_dict = {criterion: loss}

    def _accumulate(self, loss_val, mode):
        self._loss_total[mode] += loss_val
        
    def _clear(self):
        self._loss_total = {mode: 0. for mode in MODE}
        
    def _backward(self):
        self._loss_total['train'].backward()
    
    def _get_total(self, mode):
        return self._loss_total[mode].item()
        
    def backward(self):
        self._backward()
        self._clear()
    
    def __call__(self, pred, target, mode='train', *args: Any, **kwds: Any) -> Any:
        _mode = mode.lower()
        assert _mode in MODE, f"{_mode} is not defined at our mode list ({MODE})"
        for loss_key, loss_func in self.loss_func_dict.items():
            loss_val = loss_func(pred, target)
            self.loss_val_dict[_mode][loss_key].update(loss_val.item())
            self._accumulate(loss_val, _mode)

        self.loss_val_dict[_mode]['total'].update(self._get_total(_mode))
        
    def result(self, mode='train'):
        _mode = mode.lower()
        return self.loss_val_dict[_mode]

def build_losses(args):
    loss_handler = LossFactory(args)
    return loss_handler