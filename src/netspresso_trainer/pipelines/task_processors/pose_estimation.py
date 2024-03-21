from typing import Literal

import numpy as np
import torch
from loguru import logger

from .base import BaseTaskProcessor


class PoseEstimationProcessor(BaseTaskProcessor):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs):
        super(PoseEstimationProcessor, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs)

    def train_step(self, train_model, batch):
        train_model.train()
        images, keypoints = batch['pixel_values'], batch['keypoints']
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        self.optimizer.zero_grad()

        out = train_model(images)
        self.loss_factory.calc(out, target, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        pred = self.postprocessor(out)

        keypoints = keypoints.detach().cpu().numpy()
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
        eval_model.eval()
        indices, images, keypoints = batch['indices'], batch['pixel_values'], batch['keypoints']
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        out = eval_model(images)
        self.loss_factory.calc(out, target, phase='valid')

        pred = self.postprocessor(out)

        keypoints = keypoints.detach().cpu().numpy()
        if self.conf.distributed:
            pred = pred[indices != -1]
            keypoints = keypoints[indices != -1]

            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(keypoints, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [self.metric_factory.calc(g_pred, g_labels, phase='valid') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            self.metric_factory.calc(pred, keypoints, phase='valid')

        # TODO: Return gathered samples
        logs = {
            'images': images.detach().cpu().numpy(),
            'target': keypoints,
            'pred': pred
        }
        return dict(logs.items())

    def test_step(self, test_model, batch):
        test_model.eval()
        indices, images = batch['indices'], batch['pixel_values']
        images = images.to(self.devices)

        out = test_model(images)

        pred = self.postprocessor(out)

        if self.conf.distributed:
            pred = pred[indices != -1]

            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_pred = sum(gathered_pred, [])
                pred = gathered_pred

        if self.single_gpu_or_rank_zero:
            return pred

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        pass
