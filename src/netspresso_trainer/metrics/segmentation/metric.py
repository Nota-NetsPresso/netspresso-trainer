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

from ..base import BaseMetric

IGNORE_INDEX_NONE_VALUE = 255


#TODO: Temporarily added IoUMeter. MetricMeter should be more generalized.
class IoUMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_classes, name: str, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, intersection, union) -> None:
        self.intersection += intersection
        self.union += union

    def __str__(self):
        return f'{self.name} {np.nanmean(self.intersection / self.union):6.2f}'

    @property
    def avg(self) -> float:
        return np.nanmean(self.intersection / self.union)


class SegmentationMetric(BaseMetric):
    SUPPORT_METRICS: List[str] = ['iou', 'pixel_acc']

    def __init__(self, num_classes=None, ignore_index=IGNORE_INDEX_NONE_VALUE):
        # TODO: Select metrics by user
        metric_names = ['iou', 'pixel_acc']
        primary_metric = 'iou'

        assert set(metric_names).issubset(SegmentationMetric.SUPPORT_METRICS)
        super().__init__(metric_names=metric_names, primary_metric=primary_metric)
        self.ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_VALUE
        self.K = num_classes

        self.metric_meter['iou'] = IoUMeter(num_classes=self.K, name='iou')

    def intersection_and_union(self, output, target):

        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (len(output.shape) in [1, 2, 3])

        assert output.shape == target.shape
        output = output.reshape(-1)
        target = target.reshape(-1)
        output[target == self.ignore_index] = self.ignore_index
        intersection = output[output == target]
        #area_intersection = torch.histc(intersection, bins=self.K, min=0, max=self.K-1)
        #area_output = torch.histc(output, bins=self.K, min=0, max=self.K-1)
        #area_target = torch.histc(target, bins=self.K, min=0, max=self.K-1)
        area_intersection = np.histogram(intersection, bins=np.linspace(0, self.K, self.K+1))[0]
        area_output = np.histogram(output, bins=np.linspace(0, self.K, self.K+1))[0]
        area_target = np.histogram(target, bins=np.linspace(0, self.K, self.K+1))[0]
        area_union = area_output + area_target - area_intersection

        intersection, union, target, output = area_intersection, area_union, area_target, area_output

        return {
            'intersection': intersection,
            'union': union,
            'target': target,
            'output': output
        }

    def calibrate(self, pred, target, **kwargs):
        B = pred.shape[0]

        metrics = self.intersection_and_union(pred, target)
        self.metric_meter['iou'].update(metrics['intersection'], metrics['union'])
        self.metric_meter['pixel_acc'].update(sum(metrics['intersection']) / (sum(metrics['target']) + 1e-10), n=B)
