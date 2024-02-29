from typing import Literal

import numpy as np
import torch
from loguru import logger

from .base import BasePipeline


class PoseEstimationPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs):
        super(PoseEstimationPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs)

    def train_step(self, batch):
        self.model.train()
        images, keypoints = batch['pixel_values'], batch['label']
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss_factory.calc(out, target, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        pred = self.postprocessor(out)

        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(keypoints, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [self.metric_factory.calc(g_pred, g_labels, phase='train') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            self.metric_factory.calc(pred, keypoints, phase='train')

    def valid_step(self, eval_model, batch):
        pass

    def test_step(self, batch):
        pass

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        pass
