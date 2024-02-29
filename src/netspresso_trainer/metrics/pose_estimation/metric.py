from typing import List

import torch

from ..base import BaseMetric


class PoseEstimationMetric(BaseMetric):
    metric_names: List[str] = ['PCK']
    primary_metric: str = 'PCK'

    def __init__(self, **kwargs):
        super().__init__()

    def calibrate(self, pred, target, **kwargs):
        result_dict = {k: 0. for k in self.metric_names}
        return result_dict
