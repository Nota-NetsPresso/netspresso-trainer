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

from typing import Any, Dict, List

import torch

from ..utils.record import MetricMeter


class BaseMetric:
    def __init__(self, metric_name, **kwargs):
        self.metric_name = metric_name
        self.metric_meter = MetricMeter(metric_name, ':6.2f')

    def calibrate(self, pred, target, **kwargs):
        raise NotImplementedError


class MetricFactory:
    def __init__(self, task, metrics, metric_adaptor) -> None:
        self.task = task
        self.metrics = metrics
        self.metric_adaptor = metric_adaptor

    def reset_values(self):
        for phase in self.metrics:
            [metric.metric_meter.reset() for metric in self.metrics[phase]]

    def update(self, pred: torch.Tensor, target: torch.Tensor, phase: str, **kwargs: Any) -> None:
        if len(pred) == 0: # Removed dummy batch has 0 len
            return
        kwargs.update(self.metric_adaptor(pred, target))
        for metric in self.metrics[phase.lower()]:
            metric.calibrate(pred, target, **kwargs)

    def result(self, phase='train'):
        return {metric.metric_name: metric.metric_meter.avg for metric in self.metrics[phase]}

    @property
    def metric_names(self):
        return [metric.metric_name for metric in self.metrics[list(self.metrics.keys())[0]]]

    @property
    def primary_metric(self):
        return self.metric_names[0]
