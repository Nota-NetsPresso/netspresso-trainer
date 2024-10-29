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


class SegmentationMetricAdaptor:
    '''
        Adapter to process redundant operations for the metrics.
    '''
    def __init__(self, metric_names) -> None:
        self.metric_names = metric_names

    def __call__(self, predictions: List[dict], targets: List[dict]):
        return {} # Do nothing


# TODO: Unify repeated code
class mIoU(BaseMetric):
    def __init__(self, num_classes, classwise_analysis, ignore_index=IGNORE_INDEX_NONE_VALUE, **kwargs):
        metric_name = 'mIoU' # Name for logging
        super().__init__(metric_name=metric_name, num_classes=num_classes, classwise_analysis=classwise_analysis)
        self.ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_VALUE
        self.metric_meter = IoUMeter(num_classes=self.num_classes, name='iou') #TODO: Temporarily added IoUMeter.

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
        area_intersection = np.histogram(intersection, bins=np.linspace(0, self.num_classes, self.num_classes+1))[0]
        area_output = np.histogram(output, bins=np.linspace(0, self.num_classes, self.num_classes+1))[0]
        area_target = np.histogram(target, bins=np.linspace(0, self.num_classes, self.num_classes+1))[0]
        area_union = area_output + area_target - area_intersection

        intersection, union, target, output = area_intersection, area_union, area_target, area_output

        return {
            'intersection': intersection,
            'union': union,
            'target': target,
            'output': output
        }

    def calibrate(self, pred, target, **kwargs):
        metrics = self.intersection_and_union(pred, target)

        if self.classwise_analysis: # TODO: Compute in a better way
            for cls_meter, cls_intersection, cls_union in zip(self.classwise_metric_meters, metrics['intersection'], metrics['union']):
                if cls_union != 0:
                    cls_meter.update(cls_intersection, cls_union)

        self.metric_meter.update(metrics['intersection'], metrics['union'])


class PixelAccuracy(BaseMetric):
    def __init__(self, num_classes, classwise_analysis, ignore_index=IGNORE_INDEX_NONE_VALUE, **kwargs):
        metric_name = 'Pixel_acc' # Name for logging
        super().__init__(metric_name=metric_name, num_classes=num_classes, classwise_analysis=classwise_analysis)
        self.ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_VALUE

    def intersection_and_union(self, output, target):

        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (len(output.shape) in [1, 2, 3])

        assert output.shape == target.shape
        output = output.reshape(-1)
        target = target.reshape(-1)
        output[target == self.ignore_index] = self.ignore_index
        intersection = output[output == target]

        area_intersection = np.histogram(intersection, bins=np.linspace(0, self.num_classes, self.num_classes+1))[0]
        area_target = np.histogram(target, bins=np.linspace(0, self.num_classes, self.num_classes+1))[0]

        intersection, target = area_intersection, area_target

        return {
            'intersection': intersection,
            'target': target,
        }

    def calibrate(self, pred, target, **kwargs):
        B = pred.shape[0]

        metrics = self.intersection_and_union(pred, target)

        if self.classwise_analysis: # TODO: Compute in a better way
            for cls_meter, cls_intersection, cls_target in zip(self.classwise_metric_meters, metrics['intersection'], metrics['target']):
                if cls_target != 0:
                    cls_meter.update(cls_intersection, cls_target)

        self.metric_meter.update(sum(metrics['intersection']) / (sum(metrics['target']) + 1e-10), n=B)
