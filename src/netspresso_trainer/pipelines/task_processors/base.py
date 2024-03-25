from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, final

import torch.distributed as dist

NUM_SAMPLES = 16


class BaseTaskProcessor(ABC):
    def __init__(self, conf, postprocessor, devices):
        super(BaseTaskProcessor, self).__init__()
        self.conf = conf
        self.postprocessor = postprocessor
        self.devices = devices
        self.single_gpu_or_rank_zero = (not conf.distributed) or (conf.distributed and dist.get_rank() == 0)

    @abstractmethod
    def train_step(self, train_model, batch):
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, eval_model, batch):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, test_model, batch):
        raise NotImplementedError

    @abstractmethod
    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid'], metric_factory):
        raise NotImplementedError
