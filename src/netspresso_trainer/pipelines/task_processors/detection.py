from typing import Literal

import torch

from .base import BaseTaskProcessor


class DetectionProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices):
        super(DetectionProcessor, self).__init__(conf, postprocessor, devices)
        self.num_classes = 4 # TODO: Fix this
        #self.num_classes = train_dataloader.dataset.num_classes

    def train_step(self, train_model, batch, optimizer, loss_factory, metric_factory):
        train_model.train()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices),}
                   for box, label in zip(bboxes, labels)]

        targets = {'gt': targets,
                   'img_size': images.size(-1),
                   'num_classes': self.num_classes,}

        optimizer.zero_grad()

        out = train_model(images)
        loss_factory.calc(out, targets, phase='train')

        loss_factory.backward()
        optimizer.step()

        pred = self.postprocessor(out, original_shape=images[0].shape)

        if self.conf.distributed:
            gathered_bboxes = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.gather_object(bboxes, gathered_bboxes if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_bboxes = sum(gathered_bboxes, [])
                gathered_labels = sum(gathered_labels, [])
                gathered_pred = sum(gathered_pred, [])
                bboxes = gathered_bboxes
                labels = gathered_labels
                pred = gathered_pred

        if self.single_gpu_or_rank_zero:
            logs = {
                'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                        for bbox, label in zip(bboxes, labels)],
                'pred': pred
            }
            return dict(logs.items())

    def valid_step(self, eval_model, batch, loss_factory, metric_factory):
        eval_model.eval()
        indices, images, labels, bboxes = batch['indices'], batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]

        targets = {'gt': targets,
                   'img_size': images.size(-1),
                   'num_classes': self.num_classes,}

        out = eval_model(images)
        loss_factory.calc(out, targets, phase='valid')

        pred = self.postprocessor(out, original_shape=images[0].shape)

        indices = indices.numpy()
        if self.conf.distributed:
            # Remove dummy samples, they only come in distributed environment
            images = images[indices != -1]
            filtered_bboxes = []
            filtered_labels = []
            filtered_pred = []
            for idx, bool_idx in enumerate(indices != -1):
                if bool_idx:
                    filtered_bboxes.append(bboxes[idx])
                    filtered_labels.append(labels[idx])
                    filtered_pred.append(pred[idx])
            bboxes = filtered_bboxes
            labels = filtered_labels
            pred = filtered_pred

            # Gather phase
            gathered_bboxes = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.gather_object(bboxes, gathered_bboxes if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_bboxes = sum(gathered_bboxes, [])
                gathered_labels = sum(gathered_labels, [])
                gathered_pred = sum(gathered_pred, [])
                bboxes = gathered_bboxes
                labels = gathered_labels
                pred = gathered_pred

        if self.single_gpu_or_rank_zero:
            logs = {
                'images': images.detach().cpu().numpy(),
                'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                        for bbox, label in zip(bboxes, labels)],
                'pred': pred
            }
            return dict(logs.items())

    def test_step(self, test_model, batch):
        test_model.eval()
        indices, images = batch['indices'], batch['pixel_values']
        images = images.to(self.devices)

        out = test_model(images)

        pred = self.postprocessor(out, original_shape=images[0].shape)

        indices = indices.numpy()
        if self.conf.distributed:
            # Remove dummy samples, they only come in distributed environment
            filtered_pred = []
            for idx, bool_idx in enumerate(indices != -1):
                if bool_idx:
                    filtered_pred.append(pred[idx])
            pred = filtered_pred

            # Gather phase
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_pred = sum(gathered_pred, [])
                pred = gathered_pred

        if self.single_gpu_or_rank_zero:
            results = {'images': images.detach().cpu().numpy(), 'pred': pred}
            return results

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid'], metric_factory):
        if self.single_gpu_or_rank_zero:
            pred = []
            targets = []
            for output_batch in outputs:
                if len(output_batch['target']) == 0:
                    continue

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
            metric_factory.calc(pred, target=targets, phase=phase)
