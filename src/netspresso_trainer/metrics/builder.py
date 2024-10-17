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
from .registry import TASK_METRIC, PHASE_LIST

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


def build_metrics(task: str, model_conf, metrics_conf, num_classes, **kwargs) -> MetricFactory:
    # if metrics_conf is None:
    #     metrics_conf = TASK_DEFUALT_METRICS[task]
    # assert all(metric in TASK_AVAILABLE_METRICS[task] for metric in metrics_conf), \
    #     f"Available metrics for {task} are {TASK_AVAILABLE_METRICS[task]}"

    assert task in TASK_METRIC
    metric_cls = TASK_METRIC[task]

    # TODO: This code assumes there is only one loss module. Fix here later.
    if hasattr(model_conf.losses[0], 'ignore_index'):
        kwargs['ignore_index'] = model_conf.losses[0].ignore_index
    metrics = {phase: metric_cls(num_classes=num_classes, **kwargs) for phase in PHASE_LIST}

    metric_handler = MetricFactory(task, metrics)
    return metric_handler
