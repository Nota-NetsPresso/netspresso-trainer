import copy
import json
import os
from abc import ABC, abstractmethod
from ctypes import c_int
from dataclasses import asdict
from multiprocessing import Value
from pathlib import Path
from statistics import mean
from typing import Dict, List, Literal, Optional, final

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from omegaconf import DictConfig

from .base import BasePipeline
from .task_processors.base import BaseTaskProcessor
from ..loggers.base import TrainingLogger
from ..losses.builder import LossFactory
from ..metrics.builder import MetricFactory
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.fx import save_graphmodule
from ..utils.logger import yaml_for_logging
from ..utils.onnx import save_onnx
from ..utils.record import Timer, TrainingSummary
from ..utils.stats import get_params_and_macs
from ..utils.model_ema import ModelEMA

NUM_SAMPLES = 16


class EvaluationPipeline(BasePipeline):
    def __init__(
        self,
        conf: DictConfig,
        task: str,
        task_processor: BaseTaskProcessor,
        model_name: str,
        model: nn.Module,
        logger: Optional[TrainingLogger],
        timer: Timer,
        loss_factory: LossFactory,
        metric_factory: MetricFactory,
        eval_dataloader: DataLoader,
        single_gpu_or_rank_zero: bool,
    ):
        super(EvaluationPipeline, self).__init__(conf, task, task_processor, model_name, model, logger, timer)
        self.loss_factory = loss_factory
        self.metric_factory = metric_factory
        self.eval_dataloader = eval_dataloader
        self.single_gpu_or_rank_zero = single_gpu_or_rank_zero

    @final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        return True

    @property
    def valid_loss(self):
        return self.loss_factory.result('valid').get('total').avg

    @torch.no_grad()
    def evaluation(self, num_samples=NUM_SAMPLES):
        self._is_ready()
        self.timer.start_record(name='evaluation')

        num_returning_samples = 0
        returning_samples = []
        outputs = []
        for _idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            out = self.task_processor.valid_step(self.model, batch, self.loss_factory, self.metric_factory)
            if out is not None:
                outputs.append(out)
                if num_returning_samples < num_samples:
                    returning_samples.append(out)
                    num_returning_samples += len(out['pred'])
        self.task_processor.get_metric_with_all_outputs(outputs, phase='valid')

        self.timer.end_record(name='evaluation')
        time_for_evaluation = self.timer.get(name='evaluation', as_pop=False)
        self.log_end_evaluation(time_for_evaluation=time_for_evaluation, valid_samples=returning_samples)
        self.save_summary()

    def log_end_evaluation(
        self,
        time_for_evaluation: float,
        valid_samples: Optional[List] = None,
    ):
        losses = self.loss_factory.result('valid')
        metrics = self.metric_factory.result('valid')
        self.log_results(
            prefix='evaluation',
            losses=losses,
            metrics=metrics,
            elapsed_time=time_for_evaluation,
        )

    def save_summary(self):
        pass
