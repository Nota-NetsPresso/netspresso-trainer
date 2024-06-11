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

from typing import Any, Dict

import torch

from .registry import PHASE_LIST, TASK_METRIC


class MetricFactory:
    def __init__(self, task, conf_model, **kwargs) -> None:
        self.task = task
        self.conf_model = conf_model

        assert self.task in TASK_METRIC
        self.metric_cls = TASK_METRIC[self.task]

        # TODO: This code assumes there is only one loss module. Fix here later.
        if hasattr(conf_model.losses[0], 'ignore_index'):
            kwargs['ignore_index'] = conf_model.losses[0].ignore_index
        self.metrics = {phase: self.metric_cls(**kwargs) for phase in PHASE_LIST}

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


def build_metrics(task: str, conf_model, **kwargs) -> MetricFactory:
    metric_handler = MetricFactory(task, conf_model, **kwargs)
    return metric_handler
