import os
from pathlib import Path

from omegaconf import OmegaConf
import torch

from .base import BasePipeline
from ..optimizers import build_optimizer
from ..schedulers import build_scheduler
from ..utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))

MAX_SAMPLE_RESULT = 10


class ClassificationPipeline(BasePipeline):
    def __init__(self, args, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, **kwargs):
        super(ClassificationPipeline, self).__init__(args, task, model_name, model, devices,
                                                     train_dataloader, eval_dataloader, class_map, **kwargs)

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.args.training.opt,
                                         lr=self.args.training.lr0,
                                         wd=self.args.training.weight_decay,
                                         momentum=self.args.training.momentum)
        sched_args = OmegaConf.create({
            'epochs': self.args.training.epochs,
            'lr_noise': None,
            'sched': 'poly',
            'decay_rate': self.args.training.schd_power,
            'min_lr': self.args.training.lrf,  # FIXME: add hyperparameter or approve to follow `self.args.training.lrf`
            'warmup_lr': self.args.training.lr0,  # self.args.training.lr0
            'warmup_epochs': self.args.training.warmup_epochs,  # self.args.training.warmup_epochs
            'cooldown_epochs': 0,
        })
        self.scheduler, _ = build_scheduler(self.optimizer, sched_args)

    def train_step(self, batch):
        self.model.train()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss(out, target, mode='train')
        self.metric(out['pred'], target, mode='train')

        self.loss.backward()
        self.optimizer.step()

        if self.args.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        out = self.model(images)
        self.loss(out, target, mode='valid')
        self.metric(out['pred'], target, mode='valid')

        if self.args.distributed:
            torch.distributed.barrier()

    def test_step(self, batch):
        self.model.eval()
        images, _ = batch
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))
        _, pred = out['pred'].topk(1, 1, True, True)

        if self.args.distributed:
            torch.distributed.barrier()
            
        return pred