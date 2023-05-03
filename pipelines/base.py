from abc import ABC, abstractmethod
import os
from statistics import mean

import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from losses.builder import build_losses
from metrics.builder import build_metrics
from utils.search_api import ModelSearchServerHandler
from utils.timer import Timer
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

MAX_SAMPLE_RESULT = 10
START_EPOCH = 1
VALID_FREQ = 1

PROFILE_WAIT = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 10
PROFILE_REPEAT = 1


class BasePipeline(ABC):
    def __init__(self, args, task, model_name, model, devices, train_dataloader, eval_dataloader, is_online=True, profile=False):
        super(BasePipeline, self).__init__()
        self.args = args
        self.task = task
        self.model_name = model_name
        self.model = model
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.timer = Timer()

        self.loss = None
        self.metric = None
        self.optimizer = None
        self.train_logger = None

        self.ignore_index = None
        self.num_classes = None

        self.is_online = is_online
        if self.is_online:
            self.server_service = ModelSearchServerHandler(args.train.project, args.train.token)
        self.profile = profile

    def _is_ready(self):
        assert self.model is not None, "`self.model` is not defined!"
        assert self.optimizer is not None, "`self.optimizer` is not defined!"
        assert self.train_logger is not None, "`self.train_logger` is not defined!"

    @abstractmethod
    def set_train(self):
        pass

    def train(self):
        logger.info(f"Training configuration:\n{OmegaConf.to_yaml(OmegaConf.create(self.args).get('train'))}")
        logger.info("-" * 40)

        self.timer.start_record(name='train_all')
        self._is_ready()

        for num_epoch in range(START_EPOCH, self.args.train.epochs + START_EPOCH):
            self.timer.start_record(name=f'train_epoch_{num_epoch}')
            self.loss = build_losses(self.args, ignore_index=self.ignore_index)
            self.metric = build_metrics(self.args, ignore_index=self.ignore_index, num_classes=self.num_classes)

            if self.profile:
                self.profile_one_epoch()
                break
            else:
                self.train_one_epoch()  # append result in `self._one_epoch_result`

            self.timer.end_record(name=f'train_epoch_{num_epoch}')

            if num_epoch == START_EPOCH and self.is_online:  # FIXME: case for continuing training
                time_for_first_epoch = int(self.timer.get(name=f'train_epoch_{num_epoch}', as_pop=False))
                self.server_service.report_elapsed_time_for_epoch(time_for_first_epoch)

            epoch_with_valid = num_epoch % VALID_FREQ == START_EPOCH % VALID_FREQ

            if epoch_with_valid:
                self.validate()
            
            if (not self.args.distributed) or (self.args.distributed and torch.distributed.get_rank() == 0):
                self.log_end_epoch(num_epoch=num_epoch, with_valid=epoch_with_valid)
            
            logger.info("-" * 40)

        self.timer.end_record(name='train_all')
        logger.info(f"Total time: {self.timer.get(name='train_all'):.2f} s")

    def train_one_epoch(self):
        for idx, batch in enumerate(tqdm(self.train_dataloader, leave=False)):
            self.train_step(batch)
            
        self.scheduler.step()

    @torch.no_grad()
    def validate(self):
        for idx, batch in enumerate(tqdm(self.eval_dataloader, leave=False)):
            self.valid_step(batch)

    def log_end_epoch(self, num_epoch, with_valid):

        logger.info(f"Epoch: {num_epoch} / {self.args.train.epochs}")
        logger.info(f"learning rate: {self.learning_rate:.7f}")  # TODO: call before scheduler.step()
        logger.info(f"training loss: {self.train_loss:.7f}")
        logger.info(f"training metric: {[(name, value.avg) for name, value in self.metric.result('train').items()]}")

        if with_valid:
            logger.info(f"validation loss: {self.valid_loss:.7f}")
            logger.info(f"validation metric: {[(name, value.avg) for name, value in self.metric.result('valid').items()]}")

        self.log_result(num_epoch, with_valid)

    @property
    def learning_rate(self):
        return mean([param_group['lr'] for param_group in self.optimizer.param_groups])

    @property
    def train_loss(self):
        return self.loss.result('train').get('total').avg

    @property
    def valid_loss(self):
        return self.loss.result('valid').get('total').avg

    @abstractmethod
    def log_result(self, num_epoch, with_valid):
        pass

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def valid_step(self, batch):
        pass

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
