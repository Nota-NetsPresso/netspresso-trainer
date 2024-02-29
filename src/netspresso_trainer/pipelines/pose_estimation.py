from typing import Literal

import numpy as np
import torch
from loguru import logger

from .base import BasePipeline


class PoseEstimationPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs):
        super(PoseEstimationPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs)

    def train_step(self, batch):
        pass

    def valid_step(self, eval_model, batch):
        pass

    def test_step(self, batch):
        pass

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        pass
