from typing import List


class BaseMetric:
    metric_names: List[str] = []

    def __init__(self, **kwargs):
        pass

    def calibrate(self, pred, target, **kwargs):
        pass
