"""
This code is modified from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts .
"""
import math
import warnings

import torch
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestartsWithCustomWarmUp(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): The number of steps that the scheduler finishes to warmup the learning rate. Default: 5.
        total_iters (int): Maximum number of iterations. Default: 5.
        warmup_bias_lr (int): Starting learning rate for warmup period. Default: 0.
        iters_per_phase (int): Number of iterations for the first restart. Originally named as `T_0`. Default: 0.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        min_lr (float, optional): Minimum learning rate. Originally named as `eta_min`. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        scheduler_conf,
    ):
        total_iters = scheduler_conf.total_iters
        iters_per_phase = scheduler_conf.iters_per_phase

        if iters_per_phase <= 0 or not isinstance(iters_per_phase, int):
            T_0_maybe = total_iters // 10 if total_iters // 10 != 0 else total_iters
            assert T_0_maybe > 0 and isinstance(T_0_maybe, int)
            logger.info(f"Original T_0 is invalid {iters_per_phase}! Prefer to set T_0 as {T_0_maybe}.")
            iters_per_phase = T_0_maybe
        #if T_mult < 1 or not isinstance(T_mult, int):
        #    raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if iters_per_phase > total_iters:
            iters_per_phase = total_iters

        self.T_0 = iters_per_phase
        self.T_i = iters_per_phase
        self.T_mult = 1 # @illian01: fix as 1 for simplicity
        self.T_i, self.remain_iters = self.get_reassigned_t_i(self.T_0, self.T_i * self.T_mult, total_iters)
        self.total_iters = total_iters
        self.eta_min = scheduler_conf.min_lr
        self.T_cur = -1 # @illian01: fix as -1 since last_epoch is set to default
        self.warmup_bias_lr = scheduler_conf.warmup_bias_lr
        self.warmup_iters = scheduler_conf.warmup_epochs
        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning, stacklevel=2)

        if self.last_epoch >= 0 and self.last_epoch < self.warmup_iters:
            return [self.warmup_bias_lr + (float(self.last_epoch + 1) / float(max(1, self.warmup_iters))) * (base_lr - self.warmup_bias_lr)
                    for base_lr in self.base_lrs]

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    @staticmethod
    def get_reassigned_t_i(current_t_i, next_t_i, remain_epochs):
        """adjust T_i to finish at last epoch in overall schedule
        """
        if remain_epochs == 0:
            return current_t_i, 0

        assert remain_epochs >= current_t_i, f"{remain_epochs}, {current_t_i}"

        if remain_epochs < current_t_i + next_t_i:
            return remain_epochs, remain_epochs

        return current_t_i, remain_epochs

    def _step_without_given_epoch(self) -> int:
        if self.last_epoch < 0:
            epoch = 0
            return epoch

        epoch = self.last_epoch + 1
        self.T_cur = self.T_cur + 1
        if self.T_cur >= self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.remain_iters = self.remain_iters - self.T_i
            self.T_i = self.T_i * self.T_mult
            self.T_i, self.remain_iters = self.get_reassigned_t_i(self.T_i, self.T_i * self.T_mult, self.remain_iters)
        return epoch

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None:
            epoch = self._step_without_given_epoch()
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))

            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    self.T_i, _ = self.get_reassigned_t_i(self.T_i, self.T_i * self.T_mult, self.total_iters - epoch)
            else:
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
