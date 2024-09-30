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

import copy
import json
from ctypes import c_int
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Literal, Optional, final

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..loggers.base import TrainingLogger
from ..losses.builder import LossFactory
from ..metrics.builder import MetricFactory
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.fx import save_graphmodule
from ..utils.logger import yaml_for_logging
from ..utils.model_ema import ModelEMA
from ..utils.onnx import save_onnx
from ..utils.record import Timer, TrainingSummary
from ..utils.stats import get_params_and_flops
from .base import BasePipeline
from .task_processors.base import BaseTaskProcessor

NUM_SAMPLES = 16


class TrainingPipeline(BasePipeline):
    def __init__(
        self,
        conf: DictConfig,
        task: str,
        task_processor: BaseTaskProcessor,
        model_name: str,
        model: nn.Module,
        logger: Optional[TrainingLogger],
        timer: Timer,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_factory: LossFactory,
        metric_factory: MetricFactory,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        single_gpu_or_rank_zero: bool,
        is_graphmodule_training: bool,
        model_ema: Optional[ModelEMA],
        start_epoch: int,
        cur_epoch: c_int,
        profile: bool,
    ):
        super(TrainingPipeline, self).__init__(conf, task, task_processor, model_name, model, logger, timer)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_factory = loss_factory
        self.metric_factory = metric_factory
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.single_gpu_or_rank_zero = single_gpu_or_rank_zero
        self.is_graphmodule_training = is_graphmodule_training
        self.model_ema = model_ema
        self.start_epoch = start_epoch
        self.cur_epoch = cur_epoch
        self.profile = profile  # TODO: provide torch_tb_profiler for training

        self.training_history: Dict[int, Dict[
            Literal['train_losses', 'valid_losses', 'train_metrics', 'valid_metrics'], Dict[str, float]
        ]] = {}

        # TODO: These will be removed
        self.save_optimizer_state = True

    @final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        """Append here if you need more assertion checks!"""
        assert self.conf.logging.save_checkpoint_epoch % self.conf.logging.validation_epoch == 0, \
            "`save_checkpoint_epoch` should be the multiplier of `validation_epoch`."
        return True

    def epoch_with_valid_logging(self, epoch: int):
        validation_freq = self.conf.logging.validation_epoch
        last_epoch = epoch == self.conf.training.epochs
        return (epoch % validation_freq == 1 % validation_freq) or last_epoch

    def epoch_with_checkpoint_saving(self, epoch: int):
        checkpoint_freq = self.conf.logging.save_checkpoint_epoch
        last_epoch = epoch == self.conf.training.epochs
        return (epoch % checkpoint_freq == 1 % checkpoint_freq) or last_epoch

    @property
    def learning_rate(self):
        return mean([param_group['lr'] for param_group in self.optimizer.param_groups])

    @property
    def train_loss(self):
        return self.loss_factory.result('train').get('total').avg

    @property
    def valid_loss(self):
        return self.loss_factory.result('valid').get('total').avg

    def train(self):
        if self.single_gpu_or_rank_zero:
            logger.debug(f"Training configuration:\n{yaml_for_logging(self.conf)}")
            logger.info("-" * 40)

        self.timer.start_record(name='train_all')
        self._is_ready()

        num_epoch = -1
        try:
            for num_epoch in range(self.start_epoch, self.conf.training.epochs + 1):
                self.timer.start_record(name=f'train_epoch_{num_epoch}')
                self.loss_factory.reset_values()
                self.metric_factory.reset_values()
                self.cur_epoch.value = num_epoch

                self.train_one_epoch(epoch=num_epoch)

                with_valid_logging = self.epoch_with_valid_logging(num_epoch)
                with_checkpoint_saving = self.epoch_with_checkpoint_saving(num_epoch)
                # FIXME: multi-gpu sample counting & validation
                valid_samples = self.validate() if with_valid_logging else None

                self.timer.end_record(name=f'train_epoch_{num_epoch}')
                time_for_epoch = self.timer.get(name=f'train_epoch_{num_epoch}', as_pop=False)

                if self.single_gpu_or_rank_zero:
                    self.log_end_epoch(epoch=num_epoch,
                                       time_for_epoch=time_for_epoch,
                                       valid_samples=valid_samples,
                                       valid_logging=with_valid_logging)
                    self.save_summary()
                    if with_checkpoint_saving:
                        assert with_valid_logging
                        self.save_checkpoint(epoch=num_epoch)
                    logger.info("-" * 40)

                self.scheduler.step()  # call after reporting the current `learning_rate`

            self.timer.end_record(name='train_all')
            total_train_time = self.timer.get(name='train_all', as_pop=False)
            logger.info(f"Total time: {total_train_time:.2f} s")

            if self.single_gpu_or_rank_zero:
                self.logger.log_end_of_traning(final_metrics={'time_for_last_epoch': time_for_epoch})
                self.save_best()
                self.save_summary(end_training=True, status="success")
        except KeyboardInterrupt as e:
            # TODO: add independent procedure for KeyboardInterupt
            logger.error("Keyboard interrupt detected! Try saving the current checkpoint...")
            if self.single_gpu_or_rank_zero:
                self.save_checkpoint(epoch=num_epoch)
                self.save_best()
                self.save_summary(status="stop")
            raise e
        except Exception as e:
            logger.error("Error occurred! Try saving the current checkpoint...")
            if self.single_gpu_or_rank_zero:
                self.save_checkpoint(epoch=num_epoch)
                self.save_best()
                self.save_summary(status="error", error_stats=str(e))
            logger.error(str(e))
            raise e

    def train_one_epoch(self, epoch):
        outputs = []
        for _idx, batch in enumerate(tqdm(self.train_dataloader, leave=False)):
            out = self.task_processor.train_step(self.model, batch, self.optimizer, self.loss_factory, self.metric_factory)
            if self.model_ema:
                self.model_ema.update(model=self.model.module if hasattr(self.model, 'module') else self.model)
            outputs.append(out)
        self.task_processor.get_metric_with_all_outputs(outputs, phase='train', metric_factory=self.metric_factory)

    @torch.no_grad()
    def validate(self, num_samples=NUM_SAMPLES):
        num_returning_samples = 0
        returning_samples = []
        outputs = []
        eval_model = self.model_ema.ema_model if self.model_ema else self.model
        for _idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            out = self.task_processor.valid_step(eval_model, batch, self.loss_factory, self.metric_factory)
            if out is not None:
                outputs.append(out)
                if num_returning_samples < num_samples:
                    returning_samples.append(out)
                    num_returning_samples += len(out['pred'])
        self.task_processor.get_metric_with_all_outputs(outputs, phase='valid', metric_factory=self.metric_factory)
        return returning_samples

    def log_end_epoch(
        self,
        epoch: int,
        time_for_epoch: float,
        valid_samples: Optional[List] = None,
        valid_logging: bool = False,
    ):
        train_losses = self.loss_factory.result('train')
        train_metrics = self.metric_factory.result('train')
        self.log_results(prefix='training', epoch=epoch, losses=train_losses, metrics=train_metrics,
                         learning_rate=self.learning_rate, elapsed_time=time_for_epoch)

        if valid_logging:
            valid_losses = self.loss_factory.result('valid') if valid_logging else None
            valid_metrics = self.metric_factory.result('valid') if valid_logging else None
            self.log_results(prefix='validation', epoch=epoch, samples=valid_samples, losses=valid_losses, metrics=valid_metrics)

        summary_record = {'train_losses': train_losses, 'train_metrics': train_metrics}
        if valid_logging:
            summary_record.update({'valid_losses': valid_losses, 'valid_metrics': valid_metrics})
        self.training_history.update({epoch: summary_record})

    def save_checkpoint(self, epoch: int):
        if self.model_ema:
            model = self.model_ema.ema_model
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'deploy'):
            model.deploy()
        save_dtype = model.save_dtype

        if save_dtype == torch.float16:
            model = copy.deepcopy(model).type(save_dtype)
        logging_dir = self.logger.result_dir
        model_path = Path(logging_dir) / f"{self.task}_{self.model_name}_epoch_{epoch}.ext"
        optimizer_path = Path(logging_dir) / f"{self.task}_{self.model_name}_epoch_{epoch}_optimzer.pth"

        if self.save_optimizer_state:
            optimizer = self.optimizer.module if hasattr(self.optimizer, 'module') else self.optimizer
            save_dict = {'optimizer': optimizer.state_dict(), 'last_epoch': epoch}
            torch.save(save_dict, optimizer_path)
            logger.debug(f"Optimizer state saved at {str(optimizer_path)}")

        if self.is_graphmodule_training:
            # Just save graphmodule checkpoint
            torch.save(model, model_path.with_suffix(".pt"))
            logger.debug(f"PyTorch FX model saved at {str(model_path.with_suffix('.pt'))}")
            return
        pytorch_model_state_dict_path = model_path.with_suffix(".safetensors")
        save_checkpoint(model.state_dict(), pytorch_model_state_dict_path)
        logger.debug(f"PyTorch model saved at {str(pytorch_model_state_dict_path)}")

    def save_best(self):
        valid_losses = {epoch: record['valid_losses'].get('total') for epoch, record in self.training_history.items()
                if 'valid_losses' in record}
        if not valid_losses:
            return # No validation loss recorded
        best_epoch = min(valid_losses, key=valid_losses.get)

        logging_dir = self.logger.result_dir

        best_checkpoint_path = Path(logging_dir) / f"{self.task}_{self.model_name}_epoch_{best_epoch}.ext"
        best_model_save_path = Path(logging_dir) / f"{self.task}_{self.model_name}_best.ext"

        model = self.model.module if hasattr(self.model, 'module') else self.model
        save_dtype = model.save_dtype
        best_model_to_save = copy.deepcopy(model)
        if hasattr(best_model_to_save, 'deploy'):
            best_model_to_save.deploy()

        if self.is_graphmodule_training:
            best_model_to_save.load_state_dict(load_checkpoint(best_checkpoint_path.with_suffix('.pt')).state_dict())
            save_onnx(best_model_to_save, best_model_save_path.with_suffix(".onnx"), sample_input=self.sample_input.type(save_dtype), opset_version=self.conf.logging.onnx_export_opset)
            logger.info(f"ONNX model converting and saved at {str(best_model_save_path.with_suffix('.onnx'))}")
            torch.save(best_model_to_save, best_model_save_path.with_suffix(".pt"))
            logger.info(f"Best model saved at {str(best_model_save_path.with_suffix('.pt'))}")
            return

        best_model_to_save.load_state_dict(load_checkpoint(best_checkpoint_path.with_suffix('.safetensors')))
        pytorch_best_model_state_dict_path = best_model_save_path.with_suffix(".safetensors")
        save_checkpoint(best_model_to_save.state_dict(), pytorch_best_model_state_dict_path)
        logger.info(f"Best model saved at {str(pytorch_best_model_state_dict_path)}")

        try:
            save_onnx(best_model_to_save, best_model_save_path.with_suffix(".onnx"), sample_input=self.sample_input.type(save_dtype), opset_version=self.conf.logging.onnx_export_opset)
            logger.info(f"ONNX model converting and saved at {str(best_model_save_path.with_suffix('.onnx'))}")

            save_graphmodule(best_model_to_save, (best_model_save_path.parent / f"{best_model_save_path.stem}_fx").with_suffix(".pt"))
            logger.info(f"PyTorch FX model tracing and saved at {str(best_model_save_path.with_suffix('.pt'))}")
        except Exception as e:
            logger.error(e)
            pass

    def save_summary(self, end_training=False, status="", error_stats=""):
        training_summary = TrainingSummary(
            total_epoch=self.conf.training.epochs,
            train_losses={epoch: record['train_losses'].get('total') for epoch, record in self.training_history.items()},
            valid_losses={epoch: record['valid_losses'].get('total') for epoch, record in self.training_history.items()
                          if 'valid_losses' in record},
            train_metrics={epoch: record['train_metrics'] for epoch, record in self.training_history.items()},
            valid_metrics={epoch: record['valid_metrics'] for epoch, record in self.training_history.items()
                           if 'valid_metrics' in record},
            metrics_list=self.metric_factory.metric_names,
            primary_metric=self.metric_factory.primary_metric,
        )
        if end_training:
            total_train_time = self.timer.get(name='train_all', as_pop=True)
            flops, params = get_params_and_flops(self.model, self.sample_input.float())
            logger.info(f"[Model stats] | Sample input: {tuple(self.sample_input.shape)} | Params: {(params/1e6):.2f}M | FLOPs: {(flops/1e9):.2f}G")
            training_summary.total_train_time = total_train_time
            training_summary.flops = flops
            training_summary.params = params

        training_summary.status = status
        training_summary.error_stats = error_stats

        logging_dir = self.logger.result_dir
        summary_path = Path(logging_dir) / "training_summary.json"

        with open(summary_path, 'w') as f:
            json.dump(asdict(training_summary), f, indent=4)
        logger.info(f"Model training summary saved at {str(summary_path)}")

    def profile_one_epoch(self):
        PROFILE_WAIT = 1
        PROFILE_WARMUP = 1
        PROFILE_ACTIVE = 10
        PROFILE_REPEAT = 1
        #_ = torch.ones(1).to(self.devices)
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=PROFILE_WAIT,
                                             warmup=PROFILE_WARMUP,
                                             active=PROFILE_ACTIVE,
                                             repeat=PROFILE_REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            for idx, batch in enumerate(self.train_dataloader):
                if idx >= (PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE) * PROFILE_REPEAT:
                    break
                self.task_processor.train_step(batch)
                prof.step()
