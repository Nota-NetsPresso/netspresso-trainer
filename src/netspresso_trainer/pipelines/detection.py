from typing import Literal

import numpy as np
import torch
from loguru import logger

from .base import BasePipeline


class DetectionPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices,
                 train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs):
        super(DetectionPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, logging_dir, **kwargs)
        self.num_classes = train_dataloader.dataset.num_classes

    def train_step(self, batch):
        self.model.train()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices),}
                   for box, label in zip(bboxes, labels)]

        targets = {'gt': targets,
                   'img_size': images.size(-1),
                   'num_classes': self.num_classes,}

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss_factory.calc(out, targets, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        pred = self.postprocessor(out, original_shape=images[0].shape)

        if self.conf.distributed:
            torch.distributed.barrier()

        logs = {
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': pred
        }
        return dict(logs.items())

    def valid_step(self, batch):
        self.model.eval()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]

        targets = {'gt': targets,
                   'img_size': images.size(-1),
                   'num_classes': self.num_classes,}

        self.optimizer.zero_grad()

        out = self.model(images)
        self.loss_factory.calc(out, targets, phase='valid')

        pred = self.postprocessor(out, original_shape=images[0].shape)

        if self.conf.distributed:
            torch.distributed.barrier()

        logs = {
            'images': images.detach().cpu().numpy(),
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': pred
        }
        return dict(logs.items())

    def test_step(self, batch):
        self.model.eval()
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))

        pred = self.postprocessor(out, original_shape=images[0].shape)

        results = pred
        return results

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        pred = []
        targets = []
        for output_batch in outputs:
            for detection, class_idx in output_batch['target']:
                target_on_image = {}
                target_on_image['boxes'] = detection
                target_on_image['labels'] = class_idx
                targets.append(target_on_image)

            for detection, class_idx in output_batch['pred']:
                pred_on_image = {}
                pred_on_image['post_boxes'] = detection[..., :4]
                pred_on_image['post_scores'] = detection[..., -1]
                pred_on_image['post_labels'] = class_idx
                pred.append(pred_on_image)
        self.metric_factory.calc(pred, target=targets, phase=phase)
