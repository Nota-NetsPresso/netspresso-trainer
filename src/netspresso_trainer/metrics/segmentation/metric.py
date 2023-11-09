from typing import List

import numpy as np
import torch

from ...utils.record import AverageMeter
from ..base import BaseMetric

IGNORE_INDEX_NONE_VALUE = -100


class SegmentationMetric(BaseMetric):
    metric_names: List[str] = ['iou', 'pixel_acc']
    primary_metric: str = 'iou'

    def __init__(self, num_classes=None, ignore_index=IGNORE_INDEX_NONE_VALUE):
        super().__init__()
        self.ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_VALUE
        self.K = num_classes

    def intersection_and_union_gpu(self, output, target):

        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])

        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == self.ignore_index] = self.ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection, bins=self.K, min=0, max=self.K-1)
        area_output = torch.histc(output, bins=self.K, min=0, max=self.K-1)
        area_target = torch.histc(target, bins=self.K, min=0, max=self.K-1)
        area_union = area_output + area_target - area_intersection

        intersection, union, target, output = \
            area_intersection.cpu().numpy(), area_union.cpu().numpy(), area_target.cpu().numpy(), area_output.cpu().numpy()

        return {
            'intersection': intersection,
            'union': union,
            'target': target,
            'output': output
        }

    def calibrate(self, pred, target, **kwargs):
        result_dict = {k: AverageMeter(k) for k in self.metric_names}
        B = pred.size(0)

        metrics = self.intersection_and_union_gpu(pred, target)
        result_dict['iou'].update(sum(metrics['intersection']) / (sum(metrics['union']) + 1e-10), n=B)
        result_dict['pixel_acc'].update(sum(metrics['intersection']) / (sum(metrics['target']) + 1e-10), n=B)

        return {k: v.avg for k, v in result_dict.items()}
