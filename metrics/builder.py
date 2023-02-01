from typing import Any

from metrics.classification.accuracy import accuracy_topk
from utils.common import AverageMeter

MODE = ['train', 'valid', 'test']

class MetricFactory:
    def __init__(self, args) -> None:
        self.metric_func_dict = dict()
        self._build_metric(args)
        self.metric_val_dict = {
            _mode: {
                metric_key: AverageMeter(metric_key, ':6.2f')
                for metric_key in self.metric_func_dict.keys()
            }
            for _mode in MODE
        }
                
        
    def _build_metric(self, args):
        # TODO: decide metrics by arguments
        self.metric_func_dict['Acc@1'] = lambda pred, target: accuracy_topk(pred, target, topk=(1, ))[0]
        self.metric_func_dict['Acc@5'] = lambda pred, target: accuracy_topk(pred, target, topk=(5, ))[0]
        
    def _clear(self):
        for avg_meter in self.loss_val_dict.values():
            avg_meter.reset()
        
    def clear(self):
        self._clear()
    
    def __call__(self, pred, target, mode='train', *args: Any, **kwds: Any) -> Any:
        _mode = mode.lower()
        assert _mode in MODE, f"{_mode} is not defined at our mode list ({MODE})"
        for metric_key, metric_func in self.metric_func_dict.items():
            metric_val = metric_func(pred, target)
            self.metric_val_dict[_mode][metric_key].update(metric_val.item())

    def result(self, mode='train'):
        _mode = mode.lower()
        return self.metric_val_dict[_mode]

def build_metrics(args):
    metric_handler = MetricFactory(args)
    return metric_handler