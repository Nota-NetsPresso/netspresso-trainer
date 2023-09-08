from typing import Any, Dict

import torch

from .registry import TASK_METRIC
from ..utils.record import AverageMeter

MODE = ['train', 'valid', 'test']
IGNORE_INDEX_NONE_AS = -100  # following PyTorch preference


class MetricFactory:
    def __init__(self, task, conf_model, **kwargs) -> None:
        self.task = task
        self.conf_model = conf_model

        assert self.task in TASK_METRIC
        metric_cls = TASK_METRIC[self.task]

        self.metric_meter_dict: Dict[str, Dict[str, AverageMeter]] = {
            mode: {
                metric_key: AverageMeter(metric_key, ':6.2f')
                for metric_key in metric_cls.metric_names
            }
            for mode in MODE
        }

        self.metric_fn = metric_cls(**kwargs)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mode='train', *args: Any, **kwargs: Any) -> Any:

        metric_result_dict = self.metric_fn.calibrate(pred, target)
        mode = mode.lower()
        assert mode in MODE, f"{mode} is not defined at our mode list ({MODE})"
        for metric_key in self.metric_meter_dict[mode]:
            assert metric_key in metric_result_dict
            self.metric_meter_dict[mode][metric_key].update(float(metric_result_dict[metric_key]))

    def result(self, mode='train'):
        return self.metric_meter_dict[mode.lower()]

    @property
    def metric_names(self):
        return self.metric_fn.metric_names

    @property
    def primary_metric(self):
        return self.metric_fn.primary_metric


def build_metrics(task: str, conf_model, **kwargs) -> MetricFactory:
    metric_handler = MetricFactory(task, conf_model, **kwargs)
    return metric_handler
