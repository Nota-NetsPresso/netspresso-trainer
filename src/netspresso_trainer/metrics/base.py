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

from typing import Dict, List, Any

import torch
from ..utils.record import MetricMeter
from .registry import PHASE_LIST, TASK_METRIC

TASK_AVAILABLE_METRICS = {
    'classification': ['accuracy'],
    'segmentation': ['mIoU', 'accuracy'],
    'detection': ['mAP50', 'mAP75', 'mAP50_95'],
}
TASK_DEFUALT_METRICS = {
    'classification': ['accuracy'],
    'segmentation': ['mIoU', 'accuracy'],
    'detection': ['mAP50', 'mAP75', 'mAP50_95'],
}


class BaseMetric:
    def __init__(self, metric_names, primary_metric, **kwargs):
        assert primary_metric in metric_names
        self.metric_names = metric_names
        self.primary_metric = primary_metric
        self.metric_meter = {metric_name: MetricMeter(metric_name, ':6.2f') for metric_name in metric_names}

    def calibrate(self, pred, target, **kwargs):
        pass


class MetricFactory:
    def __init__(self, task, model_conf, metrics_conf, num_classes, **kwargs) -> None:
        if metrics_conf is None:
            metrics_conf = TASK_DEFUALT_METRICS[task]
        assert all(metric in TASK_AVAILABLE_METRICS[task] for metric in metrics_conf), f"Available metrics for {task} are {TASK_AVAILABLE_METRICS[task]}"

        self.task = task
        self.model_conf = model_conf

        assert self.task in TASK_METRIC
        self.metric_cls = TASK_METRIC[self.task]

        # TODO: This code assumes there is only one loss module. Fix here later.
        if hasattr(model_conf.losses[0], 'ignore_index'):
            kwargs['ignore_index'] = model_conf.losses[0].ignore_index
        self.metrics = {phase: self.metric_cls(num_classes=num_classes, **kwargs) for phase in PHASE_LIST}

    def reset_values(self):
        for phase in PHASE_LIST:
            [meter.reset() for _, meter in self.metrics[phase].metric_meter.items()]

    def update(self, pred: torch.Tensor, target: torch.Tensor, phase: str, **kwargs: Any) -> None:
        if len(pred) == 0: # Removed dummy batch has 0 len
            return
        phase = phase.lower()
        self.metrics[phase].calibrate(pred, target)

    def result(self, phase='train'):
        return {metric_name: meter.avg for metric_name, meter in self.metrics[phase].metric_meter.items()}

    @property
    def metric_names(self):
        return self.metrics[list(self.metrics.keys())[0]].metric_names

    @property
    def primary_metric(self):
        return self.metrics[list(self.metrics.keys())[0]].primary_metric