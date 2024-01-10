import os
from typing import Literal

import torch
from loguru import logger
from omegaconf import OmegaConf

from .base import BasePipeline

MAX_SAMPLE_RESULT = 10


class ClassificationPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs):
        super(ClassificationPipeline, self).__init__(conf, task, model_name, model, devices,
                                                     train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs)

    def train_step(self, batch):
        self.model.train()
        images, labels = batch
        images = images.to(self.devices)
        labels = labels.to(self.devices)
        target = {'target': labels}

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss_factory.calc(out, target, phase='train')
        if labels.dim() > 1: # Soft label to label number
            labels = torch.argmax(labels, dim=-1)
        pred = self.postprocessor(out)
        self.metric_factory.calc(pred, labels, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, labels = batch
        images = images.to(self.devices)
        labels = labels.to(self.devices)
        target = {'target': labels}

        out = self.model(images)
        self.loss_factory.calc(out, target, phase='valid')
        if labels.dim() > 1: # Soft label to label number
            labels = torch.argmax(labels, dim=-1)
        pred = self.postprocessor(out)
        self.metric_factory.calc(pred, labels, phase='valid')

        if self.conf.distributed:
            torch.distributed.barrier()

    def test_step(self, batch):
        self.model.eval()
        images, _ = batch
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))
        pred = self.postprocessor(out, k=1)

        if self.conf.distributed:
            torch.distributed.barrier()

        return pred

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        pass
