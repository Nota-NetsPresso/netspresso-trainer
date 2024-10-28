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

from .base import MetricFactory
from .registry import METRIC_ADAPTORS, METRIC_LIST, PHASE_LIST, TASK_AVAILABLE_METRICS, TASK_DEFUALT_METRICS


def build_metrics(task: str, model_conf, metrics_conf, num_classes, **kwargs) -> MetricFactory:
    metric_names = metrics_conf.metric_names
    classwise_analysis = metrics_conf.classwise_analysis

    if metric_names is None:
        metric_names = TASK_DEFUALT_METRICS[task]
    metric_names = [m.lower() for m in metric_names]
    assert all(metric in TASK_AVAILABLE_METRICS[task] for metric in metric_names), \
        f"Available metrics for {task} are {TASK_AVAILABLE_METRICS[task]}"

    # TODO: This code assumes there is only one loss module. Fix here later.
    if hasattr(model_conf.losses[0], 'ignore_index'):
        kwargs['ignore_index'] = model_conf.losses[0].ignore_index

    metrics = {}
    for phase in PHASE_LIST:
        if phase == 'valid': # classwise_analysis is only available in valid phase
            metrics[phase] = [METRIC_LIST[name](num_classes=num_classes, classwise_analysis=classwise_analysis, **kwargs) for name in metric_names]
        else:
            metrics[phase] = [METRIC_LIST[name](num_classes=num_classes, classwise_analysis=False, **kwargs) for name in metric_names]

    metric_adaptor = METRIC_ADAPTORS[task](metric_names)

    metric_handler = MetricFactory(task, metrics, metric_adaptor, classwise_analysis)
    return metric_handler
