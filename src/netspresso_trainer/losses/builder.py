from itertools import chain
from typing import Any, Dict, Union

import torch
import torch.nn as nn

from ..utils.record import AverageMeter
from .registry import LOSS_DICT, PHASE_LIST


class LossFactory:
    def __init__(self, conf_model, **kwargs) -> None:
        self.loss_func_dict = {}
        self.loss_weight_dict = {}

        self.conf_model = conf_model
        self._build_losses()

        self._dirty_backward: bool = False
        self.loss_val_per_epoch: Dict[str, Dict[str, AverageMeter]] = {}
        self._clear_epoch_start()

    def _build_losses(self):

        for loss_element in self.conf_model.losses:
            criterion = loss_element.criterion
            loss_config = {k: v for k, v in loss_element.items() if k not in ['criterion', 'weight']}
            loss = LOSS_DICT[criterion](**loss_config)

            self.loss_func_dict.update({criterion: loss})
            loss_weight = loss_element.weight if loss_element.weight is not None else 1.0
            self.loss_weight_dict.update({criterion: loss_weight})

    def _clear_step_start(self):
        self.total_loss_for_backward: Union[int, torch.Tensor] = 0
        self._dirty_backward = False

    def _clear_epoch_start(self):
        self.loss_val_per_epoch = {
            phase: {
                loss_key: AverageMeter(loss_key, ':.4e') for loss_key in chain(self.loss_func_dict.keys(), ['total'])
            }
            for phase in PHASE_LIST
        }
        self._clear_step_start()

    def _assert_argument(self, kwargs):
        pass

    def backward(self):
        assert not self._dirty_backward
        self.total_loss_for_backward.requires_grad_(True)
        self.total_loss_for_backward.mean().backward()
        self._dirty_backward = True

    def result(self, phase='train'):
        phase = phase.lower()
        return self.loss_val_per_epoch[phase]

    def reset_values(self):
        self._clear_epoch_start()

    def calc(self, out: Dict, target: Union[torch.Tensor, Dict[str, torch.Tensor]], phase='train', **kwargs: Any) -> None:
        self.__call__(out=out, target=target, phase=phase, **kwargs)

    def __call__(self, out: Dict, target: Union[torch.Tensor, Dict[str, torch.Tensor]], phase: str, **kwargs: Any) -> None:
        phase = phase.lower()
        assert phase in PHASE_LIST, f"{phase} is not defined at our phase list ({PHASE_LIST})"

        self._assert_argument(kwargs)
        self._clear_step_start()

        for loss_key, loss_func in self.loss_func_dict.items():
            loss_val = loss_func(out, target)
            self.loss_val_per_epoch[phase][loss_key].update(loss_val.item())
            self.total_loss_for_backward += loss_val * self.loss_weight_dict[loss_key]

        self.loss_val_per_epoch[phase]['total'].update(self.total_loss_for_backward.item())


def build_losses(conf_model, **kwargs):
    loss_handler = LossFactory(conf_model, **kwargs)
    return loss_handler
