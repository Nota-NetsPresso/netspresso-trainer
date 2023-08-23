import os
import logging

from omegaconf import OmegaConf
import torch

from .base import BasePipeline
from ..optimizers import build_optimizer
from ..schedulers import build_scheduler

logger = logging.getLogger("netspresso_trainer")

MAX_SAMPLE_RESULT = 10


class ClassificationPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, **kwargs):
        super(ClassificationPipeline, self).__init__(conf, task, model_name, model, devices,
                                                     train_dataloader, eval_dataloader, class_map, **kwargs)

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.conf.training.opt,
                                         lr=self.conf.training.lr,
                                         wd=self.conf.training.weight_decay,
                                         momentum=self.conf.training.momentum)
        self.scheduler, _ = build_scheduler(self.optimizer, self.conf.training)

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

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        out = self.model(images)
        self.loss(out, target, mode='valid')
        self.metric(out['pred'], target, mode='valid')

        if self.conf.distributed:
            torch.distributed.barrier()

    def test_step(self, batch):
        self.model.eval()
        images, _ = batch
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))
        _, pred = out['pred'].topk(1, 1, True, True)

        if self.conf.distributed:
            torch.distributed.barrier()

        return pred

    def get_metric_with_all_outputs(self, outputs):
        pass
