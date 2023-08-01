import os
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

from optimizers import build_optimizer
from schedulers import build_scheduler
from pipelines.base import BasePipeline
from utils.logger import set_logger

logger = set_logger('pipelines', level=os.getenv('LOG_LEVEL', default='INFO'))


class DetectionPipeline(BasePipeline):
    def __init__(self, args, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, **kwargs):
        super(DetectionPipeline, self).__init__(args, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, **kwargs)
        self.num_classes = train_dataloader.dataset.num_classes

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

        if self.args.distributed:
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

        if self.args.distributed:
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