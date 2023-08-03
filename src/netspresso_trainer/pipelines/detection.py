import os
import logging

import torch
from omegaconf import OmegaConf

from .base import BasePipeline
from ..optimizers import build_optimizer
from ..schedulers import build_scheduler
from ..utils.logger import set_logger

logger = logging.getLogger("netspresso_trainer")


class DetectionPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, **kwargs):
        super(DetectionPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, **kwargs)
        self.num_classes = train_dataloader.dataset.num_classes

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.conf.training.opt,
                                         lr=self.conf.training.lr0,
                                         wd=self.conf.training.weight_decay,
                                         momentum=self.conf.training.momentum)
        conf_sched = OmegaConf.create({
            'epochs': self.conf.training.epochs,
            'lr_noise': None,
            'sched': 'poly',
            'decay_rate': self.conf.training.schd_power,
            'min_lr': self.conf.training.lrf,  # FIXME: add hyperparameter or approve to follow `self.conf.training.lrf`
            'warmup_lr': self.conf.training.lr0,  # self.conf.training.lr0
            'warmup_epochs': self.conf.training.warmup_epochs,  # self.conf.training.warmup_epochs
            'cooldown_epochs': 0,
        })
        self.scheduler, _ = build_scheduler(self.optimizer, conf_sched)

    def train_step(self, batch):
        self.model.train()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]
        
        self.optimizer.zero_grad()
        out = self.model(images, targets=targets)
        self.loss(out, target=targets, mode='train')

        self.loss.backward()
        self.optimizer.step()

        # TODO: metric update
        # out = {k: v.detach() for k, v in out.items()}
        # self.metric(out['pred'], target=targets, mode='train')

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]
        
        out = self.model(images, targets=targets)
        self.loss(out, target=targets, mode='valid')

        # TODO: metric update
        # self.metric(out['pred'], (labels, bboxes), mode='valid')

        if self.conf.distributed:
            torch.distributed.barrier()
        logs = {
            'images': images.detach().cpu().numpy(),
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(out['post_boxes'], out['post_labels'])],                
        }
        return {k: v for k, v in logs.items()}

    def test_step(self, batch):
        self.model.eval()
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))

        results = [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                   for bbox, label in zip(out['post_boxes'], out['post_labels'])],        

        return results