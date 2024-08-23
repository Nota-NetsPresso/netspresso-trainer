# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

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


class MetricMeter(object):
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
        self._sum += val
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
    flops: Optional[int] = None
    params: Optional[int] = None
    total_train_time: Optional[float] = None
    best_epoch: int = field(init=False)
    last_epoch: int = field(init=False)
    status: str = ""
    error_stats: str = ""

    def __post_init__(self):
        self.last_epoch = 1 if not self.train_losses else list(self.train_losses.keys())[-1] # self.train_losses is empty if error occurs before first epoch done
        self.best_epoch = self.last_epoch if not self.valid_losses else min(self.valid_losses, key=self.valid_losses.get) # self.valid_losses is empty if validation is not performed

@dataclass
class EvaluationSummary:
    losses: float
    metrics: float
    metrics_list: List[str]
    primary_metric: str
    flops: Optional[int] = None
    params: Optional[int] = None
    total_evaluation_time: Optional[float] = None
    success: bool = False


@dataclass
class InferenceSummary:
    flops: Optional[int] = None
    params: Optional[int] = None
    total_inference_time: Optional[float] = None
    success: bool = False
