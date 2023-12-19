import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

__all__ = ['AverageMeter', 'Timer', 'TrainingSummary']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self._val: float = 0.
        self._avg: float = 0.
        self._sum: float = 0.
        self._count: int = 0

    def update(self, val: Union[float, int], n: int = 1) -> None:
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    @property
    def avg(self) -> float:
        return self._avg


class TimeRecode:
    def __init__(self) -> None:
        self._start = time.time()
        self._end = None

    def end(self) -> None:
        self._end = time.time()

    @property
    def done(self) -> bool:
        return self._end is not None

    @property
    def elapsed(self) -> float:
        return self._end - self._start if self._end is not None else None


class Timer:
    """Basic timer with CRUD functions
    """

    def __init__(self) -> None:
        self.history = {}

    def start_record(self, name) -> bool:
        if name not in self.history:
            self.history[name] = TimeRecode()  # create
            return True
        return False  # fail cause alreay exists

    def _end_record(self, name) -> bool:
        if name in self.history:
            self.history[name].end()  # update
            return True  # success
        return False  # fail cause no such key

    def end_record(self, name):
        self._end_record(name)

    def get(self, name, as_pop=True) -> Optional[float]:
        if name in self.history:
            record = self.history.pop(name) if as_pop else self.history[name]  # read (+ delete)
            if not record.done:
                record.end()
                assert record.done
            return record.elapsed

        return  # no such key


TYPE_SUMMARY_RECORD = Dict[int, Union[float, Dict[str, float]]]  # {epoch: value, ...}


@dataclass
class TrainingSummary:
    total_epoch: int
    train_losses: TYPE_SUMMARY_RECORD
    valid_losses: TYPE_SUMMARY_RECORD
    train_metrics: TYPE_SUMMARY_RECORD
    valid_metrics: TYPE_SUMMARY_RECORD
    metrics_list: List[str]
    primary_metric: str
    start_epoch_at_one: bool
    macs: Optional[int] = None
    params: Optional[int] = None
    total_train_time: Optional[float] = None
    best_epoch: int = field(init=False)
    last_epoch: int = field(init=False)
    success: bool = False

    def __post_init__(self):
        self.best_epoch = min(self.valid_losses, key=self.valid_losses.get)
        self.last_epoch = list(self.train_losses.keys())[-1]
