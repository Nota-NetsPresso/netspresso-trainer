import logging
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger("netspresso_trainer")

class PolynomialLRWithWarmUp(_LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters with warmup When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): The number of steps that the scheduler finishes to warmup the learning rate. Default: 5.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, warmup_iters=5, total_iters=5, warmup_bias_lr=0, min_lr=0,
                 power=1.0, last_epoch=-1, verbose=False, **kwargs):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_bias_lr = warmup_bias_lr
        self.min_lr = min_lr
        self.power = power if power is not None else 1.0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning, stacklevel=2)

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        
        if self.last_epoch >= 0 and self.last_epoch < self.warmup_iters:
            return [self.warmup_bias_lr + (float(self.last_epoch + 1) / float(max(1, self.warmup_iters))) * (base_lr - self.warmup_bias_lr)
                    for base_lr in self.base_lrs]

        decay_steps = self.total_iters - self.warmup_iters
        decay_current_step = self.last_epoch - self.warmup_iters
        decay_factor = ((1.0 - decay_current_step / decay_steps) / (1.0 - (decay_current_step - 1) / decay_steps)) ** self.power
        return [self.min_lr + (group["lr"] - self.min_lr) * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        decay_steps = self.total_iters - self.warmup_iters        
        return [
            (
                min(
                    self.warmup_bias_lr + (base_lr - self.warmup_bias_lr) * float(self.last_epoch + 1) / float(max(1, self.warmup_iters)),
                    self.min_lr + (base_lr - self.min_lr) * (1.0 - min(self.last_epoch - self.warmup_iters, decay_steps) / decay_steps) ** self.power
                )
            )
            for base_lr in self.base_lrs
        ]