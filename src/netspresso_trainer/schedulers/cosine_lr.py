import logging
import math
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger("netspresso_trainer")

class CosineAnnealingLRWithCustomWarmUp(_LRScheduler):
    """Modified from CosineAnnealingLR in PyTorch

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): The number of steps that the scheduler finishes to warmup the learning rate. Default: 5.
        total_iters (int): Maximum number of iterations. Originally named as `T_max`. Default: 5.
        warmup_bias_lr (int): Starting learning rate for warmup period. Default: 0.
        min_lr (float): Minimum learning rate. Originally named as `eta_min`. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup_iters=5, total_iters=5, warmup_bias_lr=0, min_lr=0, last_epoch=-1, verbose=False, **kwargs):
        self.T_max = total_iters
        self.eta_min = min_lr
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning, stacklevel=2)
        
        if self.last_epoch > self.T_max:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch >= 0 and self.last_epoch < self.warmup_iters:
            return [self.warmup_bias_lr + (float(self.last_epoch + 1) / float(max(1, self.warmup_iters))) * (base_lr - self.warmup_bias_lr)
                    for base_lr in self.base_lrs]

        if self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                min(
                    self.warmup_bias_lr + (float(self.last_epoch + 1) / float(max(1, self.warmup_iters))) * (base_lr - self.warmup_bias_lr),
                    self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                )
            )
            for base_lr in self.base_lrs
        ]