from typing import Any

from metrics.classification import accuracy_topk
from metrics.segmentation import segmentation_stats
from utils.common import AverageMeter

MODE = ['train', 'valid', 'test']
IGNORE_INDEX_NONE_AS = -100  # following PyTorch preference


class MetricFactory:
    def __init__(self, args, **kwargs) -> None:
        self.metric_func_dict = dict()

        ignore_index = kwargs['ignore_index'] if 'ignore_index' in kwargs else None
        self.ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_AS
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else None
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
        task = args.train.task

        if task == 'classification':
            self.metric_func_dict['Acc@1'] = lambda pred, target: accuracy_topk(pred, target, topk=(1, ))[0]
            self.metric_func_dict['Acc@5'] = lambda pred, target: accuracy_topk(pred, target, topk=(5, ))[0]
        elif task == 'segmentation':
            assert self.num_classes is not None
            self.metric_func_dict['iou'] = \
                lambda pred, target: segmentation_stats(pred, target,
                                                        ignore_index=self.ignore_index,
                                                        num_classes=self.num_classes)['iou']
            self.metric_func_dict['pixel_acc'] = \
                lambda pred, target: segmentation_stats(pred, target,
                                                        ignore_index=self.ignore_index,
                                                        num_classes=self.num_classes)['pixel_acc']
        else:
            raise AssertionError

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


def build_metrics(args, **kwargs):
    metric_handler = MetricFactory(args, **kwargs)
    return metric_handler
