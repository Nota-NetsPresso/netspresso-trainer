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
            self.max_norm = conf.training.max_norm if hasattr(conf.training, 'max_norm') else None
        else:
            self.mixed_precision = False
            self.max_norm = None
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
