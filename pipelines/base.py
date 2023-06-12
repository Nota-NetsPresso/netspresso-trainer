from abc import ABC, abstractmethod
import os
from itertools import chain
from statistics import mean


import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from losses.builder import build_losses
from metrics.builder import build_metrics
from utils.timer import Timer
from utils.logger import set_logger, yaml_for_logging
from utils.fx import save_graphmodule
from utils.onnx import save_onnx
from loggers.builder import build_logger, START_EPOCH_ZERO_OR_ONE

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

MAX_SAMPLE_RESULT = 10
VALID_FREQ = 1

PROFILE_WAIT = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 10
PROFILE_REPEAT = 1

NUM_SAMPLES = 16


class BasePipeline(ABC):
    def __init__(self, args, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map,
                 profile=False):
        super(BasePipeline, self).__init__()
        self.args = args
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

        self.profile = profile

        self.epoch_with_valid_logging = lambda e: e % VALID_FREQ == START_EPOCH_ZERO_OR_ONE % VALID_FREQ
        self.single_gpu_or_rank_zero = (not self.args.distributed) or (self.args.distributed and torch.distributed.get_rank() == 0)

        self.train_logger = build_logger(self.args, self.task, self.model_name,
                                         step_per_epoch=self.train_step_per_epoch, class_map=class_map,
                                         num_sample_images=NUM_SAMPLES)

    # final
    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        """Append here if you need more assertion checks!"""
        return True

    @abstractmethod
    def set_train(self):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch):
        raise NotImplementedError

    def train(self):
        logger.info(f"Training configuration:\n{yaml_for_logging(self.args)}")
        logger.info("-" * 40)

        self.timer.start_record(name='train_all')
        self._is_ready()

        for num_epoch in range(START_EPOCH_ZERO_OR_ONE, self.args.training.epochs + START_EPOCH_ZERO_OR_ONE):
            self.timer.start_record(name=f'train_epoch_{num_epoch}')
            self.loss = build_losses(self.args, ignore_index=self.ignore_index)
            self.metric = build_metrics(self.args, ignore_index=self.ignore_index, num_classes=self.num_classes)

            if self.profile:
                self.profile_one_epoch()
                break
            else:
                self.train_one_epoch()  # append result in `self._one_epoch_result`

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
            self.train_logger.log_end_of_traning()

        model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model.state_dict(), 'model.pth')
        save_graphmodule(model, 'model.pt')
        save_onnx(model, 'model.onnx', sample_input=torch.randn((1, 3, self.args.training.img_size, self.args.training.img_size)))

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
