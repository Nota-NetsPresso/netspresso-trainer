from typing import Any

from metrics.classification.accuracy import accuracy_topk
from utils.common import AverageMeter

class MetricFactory:
    def __init__(self, args) -> None:
        self.metric_func_dict = dict()
        self._build_metric(args)
        self.metric_val_dict = {metric_key: AverageMeter(metric_key, ':6.2f') for metric_key in self.metric_func_dict.keys()}
        
    def _build_metric(self, args):
        self.metric_func_dict['Acc@1'] = lambda pred, target: accuracy_topk(pred, target, topk=(1, ))
        self.metric_func_dict['Acc@5'] = lambda pred, target: accuracy_topk(pred, target, topk=(5, ))
        
    def _clear(self):
        for avg_meter in self.loss_val_dict.values():
            avg_meter.reset()
        
    def clear(self):
        self._clear()
    
    def __call__(self, pred, target, *args: Any, **kwds: Any) -> Any:
        for metric_key, metric_func in self.metric_func_dict.items():
            metric_val = metric_func(pred, target)
            self.metric_val_dict[metric_key].update(metric_val.item())

    @property
    def result(self):
        return self.metric_val_dict

def build_metrics(args):
    metric_handler = MetricFactory(args)
    return metric_handler