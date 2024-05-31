# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import warnings

import torch
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRWithWarmUp(_LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters with warmup When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): The number of steps that the scheduler finishes to warmup the learning rate.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
        power (int): The power of the polynomial.
    """
    def __init__(
        self,
        optimizer,
        scheduler_conf,
        training_epochs,
    ):
        self.warmup_iters = scheduler_conf.warmup_epochs
        self.total_iters = scheduler_conf.end_epoch
        self.warmup_bias_lr = scheduler_conf.warmup_bias_lr
        self.min_lr = scheduler_conf.min_lr
        self.power = scheduler_conf.power
        super().__init__(optimizer)

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
