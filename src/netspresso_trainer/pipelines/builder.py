from ctypes import c_int
from multiprocessing import Value
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from .registry import SUPPORTING_TASK_LIST, TASK_PROCESSOR, PIPELINES
from ..postprocessors import build_postprocessor
from ..loggers import build_logger
from ..utils.model_ema import build_ema
from ..utils.record import Timer
from ..optimizers import build_optimizer
from ..schedulers import build_scheduler
from ..losses import build_losses
from ..metrics import build_metrics
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


def build_pipeline(pipeline_type, conf, task, model_name, model, devices,
                   train_dataloader, eval_dataloader, class_map, logging_dir,
                   is_graphmodule_training, profile=False):
    assert task in SUPPORTING_TASK_LIST, f"No such task! (task: {task})"

    # Build task processor
    postprocessor = build_postprocessor(task, conf.model)
    task_processor = TASK_PROCESSOR[task](postprocessor, devices, conf.distributed)

    if pipeline_type == 'train':
        # Build modules for training
        optimizer = build_optimizer(model, optimizer_conf=conf.training.optimizer)
        scheduler, _ = build_scheduler(optimizer, conf.training)
        loss_factory = build_losses(conf.model, ignore_index=None)
        metric_factory = build_metrics(task, conf.model, ignore_index=None, num_classes=None)
        optimizer, scheduler, start_epoch = load_optimizer_checkpoint(conf, optimizer, scheduler)

        # Set current epoch counter and end epoch in dataloader.dataset to use in dataset.transforms
        cur_epoch = Value(c_int, start_epoch)
        train_dataloader.dataset.cur_epoch = cur_epoch
        train_dataloader.dataset.end_epoch = conf.training.epochs

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

        # Build timer
        timer = Timer()

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
        raise NotImplementedError

    elif pipeline_type == 'inference':
        raise NotImplementedError

    else:
        raise ValueError("``pipeline_type`` must be one of ['train', 'evaluation', 'inference'].")

    return pipeline
