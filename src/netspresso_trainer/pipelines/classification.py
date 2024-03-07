import os
from typing import Literal

import numpy as np
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
        indices, images, labels = batch
        images = images.to(self.devices)
        labels = labels.to(self.devices)
        target = {'target': labels}

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss_factory.calc(out, target, phase='train')
        if labels.dim() > 1: # Soft label to label number
            labels = torch.argmax(labels, dim=-1)
        pred = self.postprocessor(out)

        self.loss_factory.backward()
        self.optimizer.step()

        labels = labels.detach().cpu().numpy() # Change it to numpy before compute metric
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [self.metric_factory.calc(g_pred, g_labels, phase='train') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            self.metric_factory.calc(pred, labels, phase='train')

    def valid_step(self, eval_model, batch):
        eval_model.eval()
        indices, images, labels = batch
        images = images.to(self.devices)
        labels = labels.to(self.devices)
        target = {'target': labels}

        out = eval_model(images)
        self.loss_factory.calc(out, target, phase='valid')
        if labels.dim() > 1: # Soft label to label number
            labels = torch.argmax(labels, dim=-1)
        pred = self.postprocessor(out)

        labels = labels.detach().cpu().numpy() # Change it to numpy before compute metric
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            # Remove dummy samples, they only come in distributed environment
            pred = pred[indices != -1]
            labels = labels[indices != -1]
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [self.metric_factory.calc(g_pred, g_labels, phase='valid') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            self.metric_factory.calc(pred, labels, phase='valid')

    def test_step(self, batch):
        self.model.eval()
        indices, images, _ = batch
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))
        pred = self.postprocessor(out, k=1)

        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]

            # Remove dummy samples, they only come in distributed environment
            pred = pred[indices != -1]
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_pred = [g for g in gathered_pred]
                gathered_pred = np.concatenate(gathered_pred, axis=0)
                pred = gathered_pred
        else:
            pred = pred

        return pred

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        pass
