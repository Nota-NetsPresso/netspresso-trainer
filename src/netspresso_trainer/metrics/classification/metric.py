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

from typing import List

import numpy as np
import torch

from ..base import BaseMetric

TOPK_MAX = 20


@torch.no_grad()
def accuracy_topk(pred, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = pred.shape[-1]
    pred = pred.T
    class_num = pred.shape[0]
    correct = np.equal(pred, np.tile(target, (class_num, 1)))
    return lambda topk: correct[:min(topk, maxk)].reshape(-1).astype('float').sum(0)


class ClassificationMetricAdaptor:
    '''
        Adapter to process redundant operations for the metrics.
    '''
    def __init__(self, metric_names) -> None:
        self.metric_names = metric_names

    def __call__(self, predictions: List[dict], targets: List[dict]):
        ret = {}
        if 'top1_accuracy' in self.metric_names or 'top5_accuracy' in self.metric_names:
            topk_callable = accuracy_topk(predictions, targets)
            ret['topk_callable'] = topk_callable

        return ret

class Top1Accuracy(BaseMetric):
    def __init__(self, **kwargs):
        metric_name = 'Acc@1' # Name for logging
        super().__init__(metric_name=metric_name)

    def calibrate(self, pred, target, **kwargs):
        topk_callable = kwargs['topk_callable']

        Acc1_correct = topk_callable(topk=1)
        self.metric_meter.update(Acc1_correct, n=pred.shape[0])


class Top5Accuracy(BaseMetric):
    def __init__(self, **kwargs):
        metric_name = 'Acc@5' # Name for logging
        super().__init__(metric_name=metric_name)

    def calibrate(self, pred, target, **kwargs):
        topk_callable = kwargs['topk_callable']

        Acc5_correct = topk_callable(topk=5)
        self.metric_meter.update(Acc5_correct, n=pred.shape[0])
