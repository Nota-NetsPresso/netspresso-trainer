from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, final

import torch
import torch.distributed as dist
from loguru import logger

NUM_SAMPLES = 16


class BaseTaskProcessor(ABC):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(BaseTaskProcessor, self).__init__()
        self.conf = conf
        self.postprocessor = postprocessor
        self.devices = devices
        self.single_gpu_or_rank_zero = (not conf.distributed) or (conf.distributed and dist.get_rank() == 0)

        #TODO: Temporarily set ``mixed_precision`` as optional since this is experimental
        if hasattr(conf, 'training'):
            self.mixed_precision = conf.training.mixed_precision if hasattr(conf.training, 'mixed_precision') else False
        else:
            self.mixed_precision = False
        if self.mixed_precision:
            if self.single_gpu_or_rank_zero:
                logger.info("-" * 40)
                logger.info("Mixed precision training activated.")
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

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
