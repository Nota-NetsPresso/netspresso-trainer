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

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger

LOG_FILENAME = "result.log"

class StdOutLogger:
    def __init__(self, task, model, total_epochs=None, result_dir=None) -> None:
        super(StdOutLogger, self).__init__()
        self.task = task
        self.model_name = model
        self.total_epochs = total_epochs if total_epochs is not None else "???"

    def __call__(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        losses : Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        learning_rate: Optional[float] = None,
        elapsed_time: Optional[float] = None,
        **kwargs
    ):
        if epoch is not None and prefix == 'training':
            logger.info(f"Epoch: {epoch} / {self.total_epochs}")

        if learning_rate is not None:
            logger.info(f"learning rate: {learning_rate:.7f}")
        if elapsed_time is not None:
            logger.info(f"elapsed_time: {elapsed_time:.7f}")

        if losses is not None:
            logger.info(f"{prefix} loss: {losses['total']:.7f}")
        if metrics is not None:
            logger.info(f"{prefix} metric: {[(name, value) for name, value in metrics.items()]}")
