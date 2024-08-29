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

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Literal, Optional, final

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..loggers.base import TrainingLogger
from ..losses.builder import LossFactory
from ..metrics.builder import MetricFactory
from ..utils.record import EvaluationSummary, Timer
from ..utils.stats import get_params_and_flops
from .base import BasePipeline
from .task_processors.base import BaseTaskProcessor

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
        self.task_processor.get_metric_with_all_outputs(outputs, phase='valid', metric_factory=self.metric_factory)

        self.timer.end_record(name='evaluation')
        if self.single_gpu_or_rank_zero:
            time_for_evaluation = self.timer.get(name='evaluation', as_pop=False)
            self.log_end_evaluation(time_for_evaluation=time_for_evaluation, valid_samples=returning_samples)

    def log_end_evaluation(
        self,
        time_for_evaluation: float,
        valid_samples: Optional[List] = None,
    ):
        losses = self.loss_factory.result('valid')
        metrics = self.metric_factory.result('valid')
        self.log_results(
            prefix='evaluation',
            samples=valid_samples,
            losses=losses,
            metrics=metrics,
            elapsed_time=time_for_evaluation,
        )
        self.save_summary(losses, metrics, time_for_evaluation)

    def save_summary(self, losses, metrics, time_for_evaluation):
        flops, params = get_params_and_flops(self.model, self.sample_input.float())
        evaluation_summary = EvaluationSummary(
            losses=losses,
            metrics=metrics,
            metrics_list=self.metric_factory.metric_names,
            primary_metric=self.metric_factory.primary_metric,
            flops=flops,
            params=params,
            total_evaluation_time=time_for_evaluation,
            success=True,
        )

        logger.info(f"[Model stats] | Sample input: {tuple(self.sample_input.shape)} | Params: {(params/1e6):.2f}M | FLOPs: {(flops/1e9):.2f}G")
        logging_dir = self.logger.result_dir
        summary_path = Path(logging_dir) / "evaluation_summary.json"

        with open(summary_path, 'w') as f:
            json.dump(asdict(evaluation_summary), f, indent=4)
        logger.info(f"Model evaluation summary saved at {str(summary_path)}")
