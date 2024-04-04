from typing import Dict, List

from ..utils.record import MetricMeter


class BaseMetric:
    def __init__(self, metric_names, primary_metric, **kwargs):
        assert primary_metric in metric_names
        self.metric_names = metric_names
        self.primary_metric = primary_metric
        self.metric_meter = {metric_name: MetricMeter(metric_name, ':6.2f') for metric_name in metric_names}

    def calibrate(self, pred, target, **kwargs):
        pass
