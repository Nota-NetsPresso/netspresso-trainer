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
from ..utils.protocols import ProcessorStepOut
from ..utils.record import InferenceSummary, PredictionSummary, Timer
from ..utils.stats import get_params_and_flops
from .base import BasePipeline
from .task_processors.base import BaseTaskProcessor


class InferencePipeline(BasePipeline):
    def __init__(
        self,
        conf: DictConfig,
        task: str,
        task_processor: BaseTaskProcessor,
        model_name: str,
        model: nn.Module,
        logger: Optional[TrainingLogger],
        timer: Timer,
        test_dataloader: DataLoader,
        single_gpu_or_rank_zero: bool,
    ):
        super(InferencePipeline, self).__init__(conf, task, task_processor, model_name, model, logger, timer)
        self.test_dataloader = test_dataloader
        self.single_gpu_or_rank_zero = single_gpu_or_rank_zero

    @final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        return True

    @torch.no_grad()
    def inference(self):
        self._is_ready()
        self.timer.start_record(name='inference')

        outputs = ProcessorStepOut.empty()
        for _idx, batch in enumerate(tqdm(self.test_dataloader, leave=False)):
            out = self.task_processor.test_step(self.model, batch)
            if self.single_gpu_or_rank_zero:
                outputs['name'].extend(out['name'])
                outputs['pred'].extend(out['pred'])

        self.timer.end_record(name='inference')
        if self.single_gpu_or_rank_zero:
            time_for_inference = self.timer.get(name='inference', as_pop=False)
            self.log_end_inference(time_for_inference=time_for_inference, valid_samples=outputs)

    def log_end_inference(
        self,
        time_for_inference: float,
        valid_samples: Optional[List] = None,
    ):
        self.log_results(
            prefix='inference',
            samples=valid_samples,
            elapsed_time=time_for_inference,
        )
        predictions = self.task_processor.get_predictions(valid_samples, self.logger.class_map)
        self.save_summary(time_for_inference, predictions)

    def save_summary(self, time_for_inference, predictions):
        flops, params = get_params_and_flops(self.model, self.sample_input.float())
        inference_summary = InferenceSummary(
            flops=flops,
            params=params,
            total_inference_time=time_for_inference,
            success=True,
        )

        predictions_summary = PredictionSummary(
            predictions=predictions,
            misc=None
        )

        logger.info(f"[Model stats] | Sample input: {tuple(self.sample_input.shape)} | Params: {(params/1e6):.2f}M | FLOPs: {(flops/1e9):.2f}G")
        logging_dir = self.logger.result_dir
        summary_path = Path(logging_dir) / "inference_summary.json"
        prediction_path = Path(logging_dir) / "predictions.json"

        with open(summary_path, 'w') as f:
            json.dump(asdict(inference_summary), f, indent=4)
        logger.info(f"Model inference summary saved at {str(summary_path)}")

        with open(prediction_path, 'w') as f:
            json.dump(asdict(predictions_summary), f, indent=4)
        logger.info(f"Model predictions are saved at {str(prediction_path)}")
