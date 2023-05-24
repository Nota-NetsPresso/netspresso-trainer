import os
from pathlib import Path
from collections import deque

from omegaconf import OmegaConf
import torch
from torch.cuda.amp import autocast

from optimizers.builder import build_optimizer
from schedulers.builder import build_scheduler
from pipelines.base import BasePipeline
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

MAX_SAMPLE_RESULT = 10


class ClassificationPipeline(BasePipeline):
    def __init__(self, args, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, **kwargs):
        super(ClassificationPipeline, self).__init__(args, task, model_name, model, devices,
                                                     train_dataloader, eval_dataloader, class_map, **kwargs)
        self.one_epoch_result = deque(maxlen=MAX_SAMPLE_RESULT)

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.args.train.opt,
                                         lr=self.args.train.lr0,
                                         wd=self.args.train.weight_decay,
                                         momentum=self.args.train.momentum)
        sched_args = OmegaConf.create({
            'epochs': self.args.train.epochs,
            'lr_noise': None,
            'sched': 'poly',
            'decay_rate': self.args.train.schd_power,
            'min_lr': 0,  # FIXME: add hyperparameter or approve to follow `self.args.train.lrf`
            'warmup_lr': 0.00001, # self.args.train.lr0
            'warmup_epochs': 5, # self.args.train.warmup_epochs
            'cooldown_epochs': 0,
        })
        self.scheduler, _ = build_scheduler(self.optimizer, sched_args)

    def train_step(self, batch):
        self.model.train()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        self.optimizer.zero_grad()
        with autocast():
            out = self.model(images)
            self.loss(out, target, mode='train')
            self.metric(out['pred'], target, mode='train')

        self.loss.backward()
        self.optimizer.step()
        

        # # TODO: fn(out)
        # fn = lambda x: x
        # self.one_epoch_result.append(self.loss.result('train'))

        if self.args.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        with autocast():
            out = self.model(images)
            self.loss(out, target, mode='valid')
            self.metric(out['pred'], target, mode='valid')

        # self.one_epoch_result.append(self.loss.result('valid'))

        if self.args.distributed:
            torch.distributed.barrier()