import logging
import os

import torch
from omegaconf import OmegaConf

from .base import BasePipeline

logger = logging.getLogger("netspresso_trainer")

MAX_SAMPLE_RESULT = 10


class ClassificationPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, **kwargs):
        super(ClassificationPipeline, self).__init__(conf, task, model_name, model, devices,
                                                     train_dataloader, eval_dataloader, class_map, **kwargs)

    def train_step(self, batch):
        self.model.train()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss_factory.calc(out, target, phase='train')
        self.metric_factory.calc(out['pred'], target, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, target = batch
        images = images.to(self.devices)
        target = target.to(self.devices)

        out = self.model(images)
        self.loss_factory.calc(out, target, phase='valid')
        self.metric_factory.calc(out['pred'], target, phase='valid')

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
