import copy
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

from ..models import build_model
from ..models.utils import DetectionModelOutput, load_from_checkpoint
from ..utils.fx import save_graphmodule
from ..utils.onnx import save_onnx
from .base import BasePipeline

logger = logging.getLogger("netspresso_trainer")


class TwoStageDetectionPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, **kwargs):
        super(TwoStageDetectionPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, **kwargs)
        self.num_classes = train_dataloader.dataset.num_classes

        # Re-compose torch.fx backbone and nn.Module head
        # To load head weights, config should have head_checkpoint value.
        if kwargs['is_graphmodule_training']:
            model = build_model(conf.model, task, self.num_classes, None, conf.augmentation.img_size)
            model.backbone = self.model
            model.head = load_from_checkpoint(model.head, conf.model.head_checkpoint)
            model = model.to(device=devices)
            self.model = model

        if conf.distributed:
            self.backbone_to_train = model.module.backbone
            self.neck_to_train = model.module.neck
            self.head_to_train = model.module.head
        else:
            self.backbone_to_train = model.backbone
            self.neck = model.neck
            self.head_to_train = model.head

    def train_step(self, batch):
        self.model.train()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]

        self.optimizer.zero_grad()

        # forward to rpn
        backbone = self.backbone_to_train
        neck = self.neck_to_train
        head = self.head_to_train

        features = backbone(images)['intermediate_features']
        features = neck(features)['intermediate_features']

        features = {str(k): v for k, v in enumerate(features)}
        rpn_features = head.rpn(features, head.image_size)

        # generate proposals for training
        proposals = rpn_features['boxes']
        proposals, matched_idxs, roi_head_labels, regression_targets = head.roi_heads.select_training_samples(proposals, targets)

        # forward to roi head
        roi_features = head.roi_heads(features, proposals, head.image_size)

        # set out
        out = DetectionModelOutput()
        out.update(rpn_features)
        out.update(roi_features)
        out.update({'labels': roi_head_labels, 'regression_targets': regression_targets})

        # Compute loss
        self.loss_factory.calc(out, target=targets, phase='train')

        self.loss_factory.backward()
        self.optimizer.step()

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        bboxes = [bbox.to(self.devices) for bbox in bboxes]
        labels = [label.to(self.devices) for label in labels]
        images = images.to(self.devices)
        targets = [{"boxes": box, "labels": label} for box, label in zip(bboxes, labels)]

        out = self.model(images)

        # Compute loss
        head = self.head_to_train
        matched_idxs, roi_head_labels = head.roi_heads.assign_targets_to_proposals(out['boxes'], bboxes, labels)
        matched_gt_boxes = [bbox[idx] for idx, bbox in zip(matched_idxs, bboxes)]
        regression_targets = head.roi_heads.box_coder.encode(matched_gt_boxes, out['boxes'])
        out.update({'labels': roi_head_labels, 'regression_targets': regression_targets})
        self.loss_factory.calc(out, target=targets, phase='valid')

        if self.conf.distributed:
            torch.distributed.barrier()

        logs = {
            'images': images.detach().cpu().numpy(),
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': [(np.concatenate((bbox.detach().cpu().numpy(), confidence.detach().cpu().numpy()[..., np.newaxis]), axis=-1),
                      label.detach().cpu().numpy())
                     for bbox, confidence, label in zip(out['post_boxes'], out['post_scores'], out['post_labels'])],
        }
        return dict(logs.items())

    def test_step(self, batch):
        self.model.eval()
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))

        results = [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                   for bbox, label in zip(out['post_boxes'], out['post_labels'])],

        return results

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid']):
        # TODO: Compute metrics for train phase
        if phase == 'train':
            return

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

    def save_checkpoint(self, epoch: int):

        # Check whether the valid loss is minimum at this epoch
        valid_losses = {epoch: record['valid_losses'].get('total') for epoch, record in self.training_history.items()
                        if 'valid_losses' in record}
        best_epoch = min(valid_losses, key=valid_losses.get)
        save_best_model = best_epoch == epoch

        model = self.model.module if hasattr(self.model, 'module') else self.model
        if self.save_dtype == torch.float16:
            model = copy.deepcopy(model).type(self.save_dtype)
        result_dir = self.train_logger.result_dir
        model_path = Path(result_dir) / f"{self.task}_{self.model_name}_epoch_{epoch}.ext"
        best_model_path = Path(result_dir) / f"{self.task}_{self.model_name}_best.ext"
        optimizer_path = Path(result_dir) / f"{self.task}_{self.model_name}_epoch_{epoch}_optimzer.pth"

        if self.save_optimizer_state:
            optimizer = self.optimizer.module if hasattr(self.optimizer, 'module') else self.optimizer
            save_dict = {'optimizer': optimizer.state_dict(), 'start_epoch_at_one': self.start_epoch_at_one, 'last_epoch': epoch}
            torch.save(save_dict, optimizer_path)
            logger.debug(f"Optimizer state saved at {str(optimizer_path)}")

        if self.is_graphmodule_training:
            # Just save graphmodule checkpoint
            torch.save(model, (model_path.parent / f"{model_path.stem}_backbone").with_suffix(".pth"))
            logger.debug(f"PyTorch FX model saved at {(model_path.parent / f'{model_path.stem}_backbone').with_suffix('.pth')}")
            torch.save(model.head.state_dict(), (model_path.parent / f"{model_path.stem}_head").with_suffix(".pth"))
            logger.info(f"Detection head saved at {(model_path.parent / f'{model_path.stem}_head').with_suffix('.pth')}")
            if save_best_model:
                save_onnx(model, best_model_path.with_suffix(".onnx"), sample_input=self.sample_input.type(self.save_dtype))
                logger.info(f"ONNX model converting and saved at {str(best_model_path.with_suffix('.onnx'))}")

                torch.save(model.backbone, (model_path.parent / f"{best_model_path.stem}_backbone").with_suffix(".pt"))
                logger.info(f"Best model saved at {(model_path.parent / f'{best_model_path.stem}_backbone').with_suffix('.pt')}")
                # save head separately
                torch.save(model.head.state_dict(), (model_path.parent / f"{best_model_path.stem}_head").with_suffix(".pth"))
                logger.info(f"Detection head saved at {(model_path.parent / f'{best_model_path.stem}_head').with_suffix('.pth')}")
            return
        torch.save(model.state_dict(), model_path.with_suffix(".pth"))
        logger.debug(f"PyTorch model saved at {str(model_path.with_suffix('.pth'))}")
        if save_best_model:
            torch.save(model.state_dict(), best_model_path.with_suffix(".pth"))
            logger.info(f"Best model saved at {str(best_model_path.with_suffix('.pth'))}")

            try:
                save_onnx(model, best_model_path.with_suffix(".onnx"), sample_input=self.sample_input.type(self.save_dtype))
                logger.info(f"ONNX model converting and saved at {str(best_model_path.with_suffix('.onnx'))}")

                # fx backbone
                save_graphmodule(model.backbone, (model_path.parent / f"{best_model_path.stem}_backbone_fx").with_suffix(".pt"))
                logger.info(f"PyTorch FX model tracing and saved at {(model_path.parent / f'{best_model_path.stem}_backbone_fx').with_suffix('.pt')}")
                # save head separately
                torch.save(model.head.state_dict(), (model_path.parent / f"{best_model_path.stem}_head").with_suffix(".pth"))
                logger.info(f"Detection head saved at {(model_path.parent / f'{best_model_path.stem}_head').with_suffix('.pth')}")
            except Exception as e:
                logger.error(e)
                pass


class OneStageDetectionPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, **kwargs):
        super(OneStageDetectionPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, **kwargs)
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

        pred = self.postprocessor(out, original_shape=images[0].shape, num_classes=self.num_classes)

        if self.conf.distributed:
            torch.distributed.barrier()

        logs = {
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': [(torch.cat([p[:, :4], p[:, 5:6]], dim=-1).detach().cpu().numpy(),
                      p[:, 6].to(torch.int).detach().cpu().numpy())
                      if p is not None else (np.array([[]]), np.array([]))
                      for p in pred]
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

        pred = self.postprocessor(out, original_shape=images[0].shape, num_classes=self.num_classes)

        if self.conf.distributed:
            torch.distributed.barrier()

        logs = {
            'images': images.detach().cpu().numpy(),
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': [(torch.cat([p[:, :4], p[:, 5:6]], dim=-1).detach().cpu().numpy(),
                      p[:, 6].to(torch.int).detach().cpu().numpy())
                      if p is not None else (np.array([[]]), np.array([]))
                      for p in pred]
        }
        return dict(logs.items())

    def test_step(self, batch):
        self.model.eval()
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))

        pred = self.postprocessor(out, original_shape=images[0].shape, num_classes=self.num_classes)

        results = [(p[:, :4].detach().cpu().numpy(), p[:, 6].to(torch.int).detach().cpu().numpy())
                   if p is not None else (np.array([[]]), np.array([]))
                   for p in pred]

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
