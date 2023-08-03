from abc import ABC, abstractmethod
import os
from statistics import mean
from pathlib import Path
from typing import final
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from ..losses import build_losses
from ..metrics import build_metrics
from ..loggers import build_logger, START_EPOCH_ZERO_OR_ONE
from ..utils.record import Timer
from ..utils.logger import yaml_for_logging
from ..utils.fx import save_graphmodule
from ..utils.onnx import save_onnx

logger = logging.getLogger("netspresso_trainer")

VALID_FREQ = 1

NUM_SAMPLES = 16


class BasePipeline(ABC):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map,
                 is_graphmodule_training=False, profile=False):
        super(BasePipeline, self).__init__()
        self.conf = conf
        self.task = task
        self.model_name = model_name
        self.model = model
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_step_per_epoch = len(train_dataloader)

        self.timer = Timer()

        self.loss = None
        self.metric = None
        self.optimizer = None

        self.ignore_index = None
        self.num_classes = None

        self.profile = profile  # TODO: provide torch_tb_profiler for training
        self.is_graphmodule_training = is_graphmodule_training

        self.epoch_with_valid_logging = lambda e: e % VALID_FREQ == START_EPOCH_ZERO_OR_ONE % VALID_FREQ
        self.single_gpu_or_rank_zero = (not self.conf.distributed) or (self.conf.distributed and torch.distributed.get_rank() == 0)

        self.train_logger = build_logger(self.conf, self.task, self.model_name,
                                         step_per_epoch=self.train_step_per_epoch, class_map=class_map,
                                         num_sample_images=NUM_SAMPLES)

    @final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        """Append here if you need more assertion checks!"""
        return True
    
    def _save_checkpoint(self, model: nn.Module):
        result_dir = self.train_logger.result_dir
        model_path = Path(result_dir) / f"{self.task}_{self.model_name}.ckpt"
        
        save_onnx(model, model_path.with_suffix(".onnx"),
                    sample_input=torch.randn((1, 3, self.conf.augmentation.img_size, self.conf.augmentation.img_size)))
        
        if self.is_graphmodule_training:
            torch.save(model, model_path.with_suffix(".pt"))
        else:
            torch.save(model.state_dict(), model_path.with_suffix(".pth"))
            save_graphmodule(model, (model_path.parent / f"{model_path.stem}_fx").with_suffix(".pt"))
        
    @abstractmethod
    def set_train(self):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch):
        raise NotImplementedError
    
    @abstractmethod
    def test_step(self, batch):
        raise NotImplementedError

    def train(self):
        logger.debug(f"Training configuration:\n{yaml_for_logging(self.conf)}")
        logger.info("-" * 40)

        self.timer.start_record(name='train_all')
        self._is_ready()

        for num_epoch in range(START_EPOCH_ZERO_OR_ONE, self.conf.training.epochs + START_EPOCH_ZERO_OR_ONE):
            self.timer.start_record(name=f'train_epoch_{num_epoch}')
            self.loss = build_losses(self.conf.model, ignore_index=self.ignore_index)
            self.metric = build_metrics(self.conf.model, ignore_index=self.ignore_index, num_classes=self.num_classes)

            self.train_one_epoch()

            with_valid_logging = self.epoch_with_valid_logging(num_epoch)
            # FIXME: multi-gpu sample counting & validation
            validation_samples = self.validate() if with_valid_logging else None

            self.timer.end_record(name=f'train_epoch_{num_epoch}')
            time_for_epoch = self.timer.get(name=f'train_epoch_{num_epoch}', as_pop=False)

            if self.single_gpu_or_rank_zero:
                self.log_end_epoch(epoch=num_epoch,
                                   time_for_epoch=time_for_epoch,
                                   validation_samples=validation_samples,
                                   valid_logging=with_valid_logging)

            self.scheduler.step()  # call after reporting the current `learning_rate`
            logger.info("-" * 40)

        self.timer.end_record(name='train_all')
        total_train_time = self.timer.get(name='train_all')
        logger.info(f"Total time: {total_train_time:.2f} s")

        if self.single_gpu_or_rank_zero:
            self.train_logger.log_end_of_traning(final_metrics={'time_for_last_epoch': time_for_epoch})

            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            self._save_checkpoint(model)

    def train_one_epoch(self):
        for idx, batch in enumerate(tqdm(self.train_dataloader, leave=False)):
            self.train_step(batch)

    @torch.no_grad()
    def validate(self, num_samples=NUM_SAMPLES):
        num_returning_samples = 0
        returning_samples = []
        for idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            out = self.valid_step(batch)
            if out is not None and num_returning_samples < num_samples:
                returning_samples.append(out)
                num_returning_samples += len(out['pred'])
        return returning_samples
    
    @torch.no_grad()
    def inference(self, test_dataset):
        returning_samples = []
        for idx, batch in enumerate(tqdm(test_dataset, leave=False)):
            out = self.test_step(batch)
            returning_samples.append(out)
        return returning_samples
        
    def log_end_epoch(self, epoch, time_for_epoch, validation_samples=None, valid_logging=False):
        train_losses = self.loss.result('train')
        train_metrics = self.metric.result('train')

        valid_losses = self.loss.result('valid') if valid_logging else None
        valid_metrics = self.metric.result('valid') if valid_logging else None

        self.train_logger.update_epoch(epoch)
        self.train_logger.log(
            train_losses=train_losses,
            train_metrics=train_metrics,
            valid_losses=valid_losses,
            valid_metrics=valid_metrics,
            train_images=None,
            valid_images=validation_samples,
            learning_rate=self.learning_rate,
            elapsed_time=time_for_epoch
        )

    @property
    def learning_rate(self):
        return mean([param_group['lr'] for param_group in self.optimizer.param_groups])

    @property
    def train_loss(self):
        return self.loss.result('train').get('total').avg

    @property
    def valid_loss(self):
        return self.loss.result('valid').get('total').avg

    def profile_one_epoch(self):
        PROFILE_WAIT = 1
        PROFILE_WARMUP = 1
        PROFILE_ACTIVE = 10
        PROFILE_REPEAT = 1
        _ = torch.ones(1).to(self.devices)
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
                self.train_step(batch)
                prof.step()
