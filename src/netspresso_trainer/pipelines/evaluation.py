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
from ..utils.protocols import ProcessorStepOut
from ..utils.record import EvaluationSummary, PredictionSummary, Timer
from ..utils.stats import get_params_and_flops
from .base import BasePipeline
from .task_processors.base import BaseTaskProcessor


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
        eval_data_stats: Dict,
        single_gpu_or_rank_zero: bool,
    ):
        super(EvaluationPipeline, self).__init__(conf, task, task_processor, model_name, model, logger, timer)
        self.loss_factory = loss_factory
        self.metric_factory = metric_factory
        self.eval_dataloader = eval_dataloader
        self.eval_data_stats = eval_data_stats
        self.single_gpu_or_rank_zero = single_gpu_or_rank_zero

    @final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        return True

    @property
    def valid_loss(self):
        return self.loss_factory.result('valid').get('total').avg

    @torch.no_grad()
    def evaluation(self):
        self._is_ready()
        self.timer.start_record(name='evaluation')

        outputs = ProcessorStepOut.empty()
        for _idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            out = self.task_processor.valid_step(self.model, batch, self.loss_factory, self.metric_factory)
            if self.single_gpu_or_rank_zero:
                outputs['name'].extend(out['name'])
                outputs['pred'].extend(out['pred'])
                outputs['target'].extend(out['target'])

        self.task_processor.get_metric_with_all_outputs(outputs, phase='valid', metric_factory=self.metric_factory)

        self.timer.end_record(name='evaluation')
        if self.single_gpu_or_rank_zero:
            time_for_evaluation = self.timer.get(name='evaluation', as_pop=False)
            self.log_end_evaluation(time_for_evaluation=time_for_evaluation, valid_samples=outputs)

    def log_end_evaluation(
        self,
        time_for_evaluation: float,
        valid_samples: Optional[List] = None,
    ):
        losses = self.loss_factory.result('valid')
        metrics = self.metric_factory.result('valid')

        # TODO: Move to logger
        # If class-wise metrics, convert to class names
        if 'classwise' in metrics[list(metrics.keys())[0]]:
            tmp_metrics = {}
            for metric_name, metric in metrics.items():
                tmp_metrics[metric_name] = {'mean': metric['mean'], 'classwise': {}}
                for cls_num, score in metric['classwise'].items():
                    cls_name = self.logger.class_map[cls_num] if cls_num in self.logger.class_map else 'mean'
                    tmp_metrics[metric_name]['classwise'][f'{cls_num}_{cls_name}'] = score
            metrics = tmp_metrics

        self.log_results(
            prefix='evaluation',
            samples=valid_samples,
            losses=losses,
            metrics=metrics,
            data_stats=self.eval_data_stats,
            elapsed_time=time_for_evaluation,
        )
        predictions = self.task_processor.get_predictions(valid_samples, self.logger.class_map)
        self.save_summary(losses, metrics, predictions, time_for_evaluation)

    def save_summary(self, losses, metrics, predictions, time_for_evaluation):
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

        predictions_summary = PredictionSummary(
            predictions=predictions,
            misc=None
        )

        logger.info(f"[Model stats] | Sample input: {tuple(self.sample_input.shape)} | Params: {(params/1e6):.2f}M | FLOPs: {(flops/1e9):.2f}G")
        logging_dir = self.logger.result_dir
        summary_path = Path(logging_dir) / "evaluation_summary.json"
        prediction_path = Path(logging_dir) / "predictions.json"

        with open(summary_path, 'w') as f:
            json.dump(asdict(evaluation_summary), f, indent=4)
        logger.info(f"Model evaluation summary saved at {str(summary_path)}")

        with open(prediction_path, 'w') as f:
            json.dump(asdict(predictions_summary), f, indent=4)
        logger.info(f"Model predictions are saved at {str(prediction_path)}")
