from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

LOG_FILENAME = "result.log"

class StdOutLogger:
    def __init__(self, task, model, total_epochs=None, result_dir=None) -> None:
        super(StdOutLogger, self).__init__()
        self.task = task
        self.model_name = model
        self.total_epochs = total_epochs if total_epochs is not None else "???"

    def init_epoch(self):
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = int(value)

    def __call__(self, train_losses, train_metrics, valid_losses, valid_metrics, learning_rate, elapsed_time):
        logger.info(f"Epoch: {self._epoch} / {self.total_epochs}")

        if learning_rate is not None:
            logger.info(f"learning rate: {learning_rate:.7f}")
        if elapsed_time is not None:
            logger.info(f"elapsed_time: {elapsed_time:.7f}")
        logger.info(f"training loss: {train_losses['total']:.7f}")
        logger.info(f"training metric: {[(name, value) for name, value in train_metrics.items()]}")

        if valid_losses is not None:
            logger.info(f"validation loss: {valid_losses['total']:.7f}")
        if valid_metrics is not None:
            logger.info(f"validation metric: {[(name, value) for name, value in valid_metrics.items()]}")
