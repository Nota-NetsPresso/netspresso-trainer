from typing import Any
from itertools import chain

import torch
import torch.nn as nn

from losses.classification.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from utils.common import AverageMeter

MODE = ['train', 'valid', 'test']
IGNORE_INDEX_NONE_AS = -100  # following PyTorch preference


class LossFactory:
    def __init__(self, args, **kwargs) -> None:
        self.loss_func_dict = dict()

        ignore_index = kwargs['ignore_index'] if 'ignore_index' in kwargs else None
        self.ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_AS
        self._build_losses(args)

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

    def _build_losses(self, args):

        # TODO: multiple loss functions
        criterion = args.train.losses.criterion

        if criterion == 'cross_entropy':
            loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif criterion == 'soft_target_cross_entropy':
            loss = SoftTargetCrossEntropy()
        elif criterion == 'label_smoothing_cross_entropy':
            loss = LabelSmoothingCrossEntropy(smoothing=args.train.losses.smoothing)  # TODO: condition
        else:
            raise AssertionError(f"Unknown criterion! ({criterion})")

        self.loss_func_dict = {criterion: loss}

    def _clear(self):
        self.total_loss_for_backward = 0

    def _backward(self):
        self.total_loss_for_backward.backward()

    def backward(self):
        self._backward()
        self._clear()

    def __call__(self, pred, target, mode='train', *args: Any, **kwds: Any) -> Any:
        _mode = mode.lower()
        assert _mode in MODE, f"{_mode} is not defined at our mode list ({MODE})"
        total_loss_per_step = 0
        for loss_key, loss_func in self.loss_func_dict.items():
            loss_val = loss_func(pred, target)
            self.loss_val_per_step[_mode][loss_key] = loss_val.item()
            self.loss_val_per_epoch[_mode][loss_key].update(loss_val.item())
            total_loss_per_step += loss_val

        if mode == 'train':
            self.total_loss_for_backward = total_loss_per_step
        self.loss_val_per_step[_mode]['total'] = total_loss_per_step.item()
        self.loss_val_per_epoch[_mode]['total'].update(total_loss_per_step.item())

    def result(self, mode='train'):
        _mode = mode.lower()
        return self.loss_val_per_epoch[_mode]


def build_losses(args, **kwargs):
    loss_handler = LossFactory(args, **kwargs)
    return loss_handler
