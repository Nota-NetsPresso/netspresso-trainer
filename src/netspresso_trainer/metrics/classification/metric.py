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


class ClassificationMetric(BaseMetric):
    SUPPORT_METRICS: List[str] = ['Acc@1', 'Acc@5']

    def __init__(self, **kwargs):
        # TODO: Select metrics by user
        metric_names = ['Acc@1', 'Acc@5']
        primary_metric = 'Acc@1'

        assert set(metric_names).issubset(ClassificationMetric.SUPPORT_METRICS)
        super().__init__(metric_names=metric_names, primary_metric=primary_metric)

    def calibrate(self, pred, target, **kwargs):
        topk_callable = accuracy_topk(pred, target)

        Acc1_correct = topk_callable(topk=1)
        Acc5_correct = topk_callable(topk=5)

        self.metric_meter['Acc@1'].update(Acc1_correct, n=pred.shape[0])
        self.metric_meter['Acc@5'].update(Acc5_correct, n=pred.shape[0])
