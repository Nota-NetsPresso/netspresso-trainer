from typing import List


class BaseMetric:
    metric_names: List[str] = ['']
    primary_metric: str = ''

    def __init__(self, **kwargs):
        assert self.primary_metric in self.metric_names
        pass

    def calibrate(self, pred, target, **kwargs):
        pass
