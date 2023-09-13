import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from .base import BasePipeline

logger = logging.getLogger("netspresso_trainer")

CITYSCAPE_IGNORE_INDEX = 255  # TODO: get from configuration


class SegmentationPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, **kwargs):
        super(SegmentationPipeline, self).__init__(conf, task, model_name, model, devices,
                                                   train_dataloader, eval_dataloader, class_map, **kwargs)
        self.ignore_index = CITYSCAPE_IGNORE_INDEX
        self.num_classes = train_dataloader.dataset.num_classes

    def train_step(self, batch):
        self.model.train()
        images, target = batch['pixel_values'], batch['labels']
        images = images.to(self.devices)
        target = target.long().to(self.devices)

        if 'edges' in batch:
            bd_gt = batch['edges']
            bd_gt = bd_gt.to(self.devices)

        self.optimizer.zero_grad()
        out = self.model(images)
        if 'edges' in batch:
            self.loss_factory.calc(out, target, bd_gt=bd_gt, phase='train')
        else:
            self.loss_factory.calc(out, target, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        out = {k: v.detach() for k, v in out.items()}
        self.metric_factory.calc(out['pred'], target, phase='train')

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, target = batch['pixel_values'], batch['labels']
        images = images.to(self.devices)
        target = target.long().to(self.devices)

        if 'edges' in batch:
            bd_gt = batch['edges']
            bd_gt = bd_gt.to(self.devices)

        out = self.model(images)
        if 'edges' in batch:
            self.loss_factory.calc(out, target, bd_gt=bd_gt, phase='valid')
        else:
            self.loss_factory.calc(out, target, phase='valid')

        self.metric_factory.calc(out['pred'], target, phase='valid')

        if self.conf.distributed:
            torch.distributed.barrier()

        output_seg = torch.max(out['pred'], dim=1)[1]  # argmax

        logs = {
            'images': images.detach().cpu().numpy(),
            'target': target.detach().cpu().numpy(),
            'pred': output_seg.detach().cpu().numpy()
        }
        if 'edges' in batch:
            logs.update({
                'bd_gt': bd_gt.detach().cpu().numpy()
            })
        return dict(logs.items())

    def test_step(self, batch):
        self.model.eval()
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))

        output_seg = torch.max(out['pred'], dim=1)[1]  # argmax

        return output_seg

    def get_metric_with_all_outputs(self, outputs):
        pass
