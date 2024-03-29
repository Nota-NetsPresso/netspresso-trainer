from typing import List

import numpy as np
import torch

from ..base import BaseMetric

TOPK_MAX = 20


@torch.no_grad()
def accuracy_topk(pred, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.shape[0]
    maxk = pred.shape[-1]
    pred = pred.T
    class_num = pred.shape[0]
    correct = np.equal(pred, np.tile(target, (class_num, 1)))
    return lambda topk: correct[:min(topk, maxk)].reshape(-1).astype('float').sum(0) * 100. / batch_size


class ClassificationMetric(BaseMetric):
    metric_names: List[str] = ['Acc@1', 'Acc@5']
    primary_metric: str = 'Acc@1'

    def __init__(self, **kwargs):
        super().__init__()

    def calibrate(self, pred, target, **kwargs):
        result_dict = {k: 0. for k in self.metric_names}
        topk_callable = accuracy_topk(pred, target)

        result_dict['Acc@1'] = topk_callable(topk=1)
        result_dict['Acc@5'] = topk_callable(topk=5)
        return result_dict
