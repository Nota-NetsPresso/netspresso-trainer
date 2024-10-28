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
from bisect import bisect_right
from collections import Counter

from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epochs reaches one of the milestones.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 150
        >>> # lr = 0.005    if 150 <= epoch < 250
        >>> # lr = 0.0005   if epoch >= 250
        >>> for epoch in range(300):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(
        self,
        optimizer,
        scheduler_conf,
        training_epochs,
    ):
        self.milestones = Counter(scheduler_conf.milestones)
        self.gamma = scheduler_conf.gamma
        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning, stacklevel=2)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
