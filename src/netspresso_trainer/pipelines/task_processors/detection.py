# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from typing import Literal, Optional

import torch

from .base import BaseTaskProcessor


class DetectionProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(DetectionProcessor, self).__init__(conf, postprocessor, devices, **kwargs)
        self.num_classes = kwargs['num_classes']

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

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            if isinstance(train_model, torch.nn.parallel.DistributedDataParallel):
                if hasattr(train_model.module.head, "_training_targets"): # for RT-DETR
                    train_model.module.head.set_training_targets(targets)
            else:
                if hasattr(train_model.head, "_training_targets"):
                    train_model.head.set_training_targets(targets)
            out = train_model(images)
            loss_factory.calc(out, targets, phase='train')

        loss_factory.backward(self.grad_scaler)
        if self.max_norm:
            # Unscales the gradients of optimizer's assigned parameters in-place
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), self.max_norm)
        # optimizer's gradients are already unscaled, so scaler.step doesn't unscale them,
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

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
            metric_factory.update(pred, target=targets, phase=phase)

    def get_predictions(self, results, class_map):
        predictions = []
        if isinstance(results, list):
            for minibatch in results:
                predictions.extend(self._convert_result(minibatch, class_map))
        elif isinstance(results, dict):
            predictions.extend(self._convert_result(results, class_map))

        return predictions

    def _convert_result(self, result, class_map):
        assert "pred" in result and "images" in result
        return_preds = []
        for idx in range(len(result['pred'])):
            image = result['images'][idx:idx+1]
            height, width = image.shape[-2:]
            preds = []
            for bbox, label in zip(result['pred'][idx][0], result['pred'][idx][1]):
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                confidence_score = float(bbox[4])
                class_idx = int(label)
                name = class_map[class_idx]
                preds.append(
                    {
                        "class": class_idx,
                        "name": name,
                        "confidence_score": confidence_score,
                        "coords": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    }
                )
            return_preds.append(
                {
                    "bboxes": preds,
                    "shape": {
                        "width": width,
                        "height": height
                    }
                }
            )

        return return_preds
