from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..loggers.base import TrainingLogger
from ..utils.record import Timer
from .task_processors.base import BaseTaskProcessor


class BasePipeline(ABC):
    def __init__(
        self,
        conf: DictConfig,
        task: str,
        task_processor: BaseTaskProcessor,
        model_name: str,
        model: nn.Module,
        logger: Optional[TrainingLogger],
        timer: Timer,
    ):
        super(BasePipeline, self).__init__()
        self.conf = conf
        self.task = task
        self.task_processor = task_processor
        self.model_name = model_name
        self.save_dtype = next(model.parameters()).dtype
        self.model = model.float()
        self.logger = logger
        self.timer = timer

    @property
    def sample_input(self):
        return torch.randn((1, 3, self.conf.augmentation.img_size, self.conf.augmentation.img_size))

    def log_results(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        samples: Optional[List] = None,
        losses : Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        learning_rate: Optional[float] = None,
        elapsed_time: Optional[float] = None,
    ):
        self.logger.log(
            prefix=prefix,
            epoch=epoch,
            samples=samples,
            losses=losses,
            metrics=metrics,
            learning_rate=learning_rate,
            elapsed_time=elapsed_time
        )

    @abstractmethod
    def save_summary(self):
        raise NotImplementedError
