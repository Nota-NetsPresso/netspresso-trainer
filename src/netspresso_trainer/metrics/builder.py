from typing import Any, Dict

import torch

from ..utils.record import AverageMeter
from .registry import PHASE_LIST, TASK_METRIC


class MetricFactory:
    def __init__(self, task, conf_model, **kwargs) -> None:
        self.task = task
        self.conf_model = conf_model

        assert self.task in TASK_METRIC
        self.metric_cls = TASK_METRIC[self.task]

        self.metric_fn = self.metric_cls(**kwargs)
        self._clear_epoch_start()

    def _clear_epoch_start(self):
        self.metric_meter_dict: Dict[str, Dict[str, AverageMeter]] = {
            phase: {
                metric_key: AverageMeter(metric_key, ':6.2f')
                for metric_key in self.metric_cls.metric_names
            }
            for phase in PHASE_LIST
        }

    def reset_values(self):
        self._clear_epoch_start()

    def calc(self, pred: torch.Tensor, target: torch.Tensor, phase='train', **kwargs: Any) -> None:
        self.__call__(pred=pred, target=target, phase=phase, **kwargs)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, phase: str, **kwargs: Any) -> None:

        metric_result_dict = self.metric_fn.calibrate(pred, target)
        phase = phase.lower()
        assert phase in PHASE_LIST, f"{phase} is not defined at our phase list ({PHASE_LIST})"
        for metric_key in self.metric_meter_dict[phase]:
            assert metric_key in metric_result_dict
            self.metric_meter_dict[phase][metric_key].update(float(metric_result_dict[metric_key]))

    def result(self, phase='train'):
        return self.metric_meter_dict[phase.lower()]

    @property
    def metric_names(self):
        return self.metric_fn.metric_names

    @property
    def primary_metric(self):
        return self.metric_fn.primary_metric


def build_metrics(task: str, conf_model, **kwargs) -> MetricFactory:
    metric_handler = MetricFactory(task, conf_model, **kwargs)
    return metric_handler
