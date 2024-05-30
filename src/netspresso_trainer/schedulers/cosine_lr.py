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

import math
import warnings

import torch
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLRWithCustomWarmUp(_LRScheduler):
    """Modified from CosineAnnealingLR in PyTorch

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): The number of steps that the scheduler finishes to warmup the learning rate.
        total_iters (int): Maximum number of iterations. Originally named as `T_max`.
        warmup_bias_lr (int): Starting learning rate for warmup period.
        min_lr (float): Minimum learning rate. Originally named as `eta_min`.
    """

    def __init__(
        self,
        optimizer,
        scheduler_conf,
        training_epochs,
    ):
        if scheduler_conf.end_epoch > training_epochs:
            logger.warning('``training.scheduler.end_epoch`` is larger than ``training.epochs``. Automatically set scheduler end_epoch as training epochs.')
            scheduler_conf.end_epoch = training_epochs
        self.T_max = scheduler_conf.end_epoch
        self.eta_min = scheduler_conf.min_lr
        self.warmup_bias_lr = scheduler_conf.warmup_bias_lr
        self.warmup_iters = scheduler_conf.warmup_epochs
        super().__init__(optimizer)

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
