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
        self.model = model
        self.logger = logger
        self.timer = timer

    @property
    def sample_input(self):
        return torch.randn((1, 3, self.conf.logging.sample_input_size[0], self.conf.logging.sample_input_size[1]))

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
