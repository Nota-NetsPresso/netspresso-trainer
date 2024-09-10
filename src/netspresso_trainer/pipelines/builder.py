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

from ctypes import c_int
from multiprocessing import Value
from pathlib import Path
from typing import Dict, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..loggers import build_logger
from ..losses import build_losses
from ..metrics import build_metrics
from ..optimizers import build_optimizer
from ..postprocessors import build_postprocessor
from ..schedulers import build_scheduler
from ..utils.model_ema import build_ema
from ..utils.record import Timer
from .registry import PIPELINES, SUPPORTING_TASK_LIST, TASK_PROCESSOR
from .train import NUM_SAMPLES


def load_optimizer_checkpoint(conf, optimizer, scheduler):
    start_epoch = 1
    resume_optimizer_checkpoint = conf.model.checkpoint.optimizer_path
    if resume_optimizer_checkpoint is not None:
        resume_optimizer_checkpoint = Path(resume_optimizer_checkpoint)
        if not resume_optimizer_checkpoint.exists():
            logger.warning(f"Traning summary checkpoint path {str(resume_optimizer_checkpoint)} is not found!"
                            f"Skip loading the previous history and trainer will be started from the beginning")
            return

        optimizer_dict = torch.load(resume_optimizer_checkpoint, map_location='cpu')
        optimizer_state_dict = optimizer_dict['optimizer']
        start_epoch = optimizer_dict['last_epoch'] + 1  # Start from the next to the end of last training

        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.step(epoch=start_epoch)

        start_epoch = start_epoch
        logger.info(f"Resume training from {str(resume_optimizer_checkpoint)}. Start training at epoch: {start_epoch}")

    return optimizer, scheduler, start_epoch


def build_pipeline(
    pipeline_type: str,
    conf: DictConfig,
    task: str,
    model_name: str,
    model: nn.Module,
    devices: torch.device,
    class_map: Dict,
    logging_dir: Union[str, Path],
    is_graphmodule_training: bool,
    dataloaders: Dict[str, DataLoader],
    profile: bool = False
):
    assert task in SUPPORTING_TASK_LIST, f"No such task! (task: {task})"

    # Build task processor
    postprocessor = build_postprocessor(task, conf.model)
    task_processor = TASK_PROCESSOR[task](conf, postprocessor, devices, num_classes=len(class_map))

    # Build timer
    timer = Timer()

    if pipeline_type == 'train':
        train_dataloader: DataLoader = dataloaders['train']
        eval_dataloader: DataLoader = dataloaders['valid']

        # Build optimizer and scheduler
        optimizer = build_optimizer(model, single_task_model=conf.model.single_task_model, optimizer_conf=conf.training.optimizer)
        scheduler, _ = build_scheduler(optimizer, conf.training)
        optimizer, scheduler, start_epoch = load_optimizer_checkpoint(conf, optimizer, scheduler)

        # Set current epoch counter and end epoch in dataloader.dataset to use in dataset.transforms
        cur_epoch = Value(c_int, start_epoch)
        train_dataloader.dataset.cur_epoch = cur_epoch
        train_dataloader.dataset.end_epoch = conf.training.epochs

        # Build loss and metric modules
        loss_factory = build_losses(conf.model, cur_epoch=cur_epoch)
        metric_factory = build_metrics(task, conf.model, num_classes=train_dataloader.dataset.num_classes)

        # Set model EMA
        model_ema = None
        if conf.training.ema:
            model_ema = build_ema(model=model.module if hasattr(model, 'module') else model, conf=conf)

        # Build logger
        single_gpu_or_rank_zero = (not conf.distributed) or (conf.distributed and dist.get_rank() == 0)
        train_step_per_epoch = len(train_dataloader)
        train_logger = None
        if single_gpu_or_rank_zero:
            train_logger = build_logger(conf, task, model_name,
                                        step_per_epoch=train_step_per_epoch,
                                        class_map=class_map,
                                        num_sample_images=NUM_SAMPLES,
                                        result_dir=logging_dir,)

        # Build pipeline
        pipeline = PIPELINES[pipeline_type](conf=conf,
                                            task=task,
                                            task_processor=task_processor,
                                            model_name=model_name,
                                            model=model,
                                            logger=train_logger,
                                            timer=timer,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            loss_factory=loss_factory,
                                            metric_factory=metric_factory,
                                            train_dataloader=train_dataloader,
                                            eval_dataloader=eval_dataloader,
                                            single_gpu_or_rank_zero=single_gpu_or_rank_zero,
                                            is_graphmodule_training=is_graphmodule_training,
                                            model_ema=model_ema,
                                            start_epoch=start_epoch,
                                            cur_epoch=cur_epoch,
                                            profile=profile)

    elif pipeline_type == 'evaluation':
        eval_dataloader: DataLoader = dataloaders['test']

        # Build modules for evaluation
        loss_factory = build_losses(conf.model)
        metric_factory = build_metrics(task, conf.model, num_classes=eval_dataloader.dataset.num_classes)

        # Build logger
        single_gpu_or_rank_zero = (not conf.distributed) or (conf.distributed and dist.get_rank() == 0)
        eval_logger = None
        if single_gpu_or_rank_zero:
            eval_logger = build_logger(conf, task, model_name,
                                       step_per_epoch=0,
                                       class_map=class_map,
                                       num_sample_images=NUM_SAMPLES,
                                       result_dir=logging_dir,)
        # Build pipeline
        pipeline = PIPELINES[pipeline_type](conf=conf,
                                            task=task,
                                            task_processor=task_processor,
                                            model_name=model_name,
                                            model=model,
                                            logger=eval_logger,
                                            timer=timer,
                                            loss_factory=loss_factory,
                                            metric_factory=metric_factory,
                                            eval_dataloader=eval_dataloader,
                                            single_gpu_or_rank_zero=single_gpu_or_rank_zero,)

    elif pipeline_type == 'inference':
        test_dataloader: DataLoader = dataloaders['test']

        # Build logger
        single_gpu_or_rank_zero = (not conf.distributed) or (conf.distributed and dist.get_rank() == 0)
        eval_logger = None
        if single_gpu_or_rank_zero:
            eval_logger = build_logger(conf, task, model_name,
                                       step_per_epoch=0,
                                       class_map=class_map,
                                       num_sample_images=NUM_SAMPLES,
                                       result_dir=logging_dir,)
        # Build pipeline
        pipeline = PIPELINES[pipeline_type](conf=conf,
                                            task=task,
                                            task_processor=task_processor,
                                            model_name=model_name,
                                            model=model,
                                            logger=eval_logger,
                                            timer=timer,
                                            test_dataloader=test_dataloader,
                                            single_gpu_or_rank_zero=single_gpu_or_rank_zero,)

    else:
        raise ValueError("``pipeline_type`` must be one of ['train', 'evaluation', 'inference'].")

    return pipeline
