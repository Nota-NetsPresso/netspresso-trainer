from typing import Any

import torch
import torch.nn as nn

from losses.classification.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from utils.common import AverageMeter

class LossFactory:
    def __init__(self, args) -> None:
        self.loss_func_dict = dict()
        self._build_losses(args)
        self.loss_val_dict = {loss_key: AverageMeter(loss_key, ':.4e') for loss_key in self.loss_func_dict.keys()}
        self.loss_val_dict.update({'total': AverageMeter('total', ':.4e')})
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

    def _accumulate(self, loss_val):
        self._loss_total += loss_val
        
    def _clear(self):
        self._loss_total = 0.
        
    def _backward(self):
        self._loss_total.backward()
        
    def backward(self):
        self._backward()
        self._clear()
    
    def __call__(self, pred, target, *args: Any, **kwds: Any) -> Any:
        for loss_key, loss_func in self.loss_func_dict.items():
            loss_val = loss_func(pred, target)
            self.loss_val_dict[loss_key].update(loss_val.item())
            self._accumulate(loss_val)

        self.loss_val_dict['total'].update(self._loss_total.item())
        
    @property
    def result(self):
        return self.loss_val_dict

def build_losses(args):
    loss_handler = LossFactory(args)
    return loss_handler