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

import cv2
import numpy as np
import torch

from netspresso_trainer.models.utils import set_training_targets

from .base import BaseTaskProcessor


class SegmentationProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(SegmentationProcessor, self).__init__(conf, postprocessor, devices, **kwargs)

    def train_step(self, train_model, batch, optimizer, loss_factory, metric_factory):
        train_model.train()
        batch['indices']
        images = batch['pixel_values'].to(self.devices)
        labels = batch['labels'].long().to(self.devices)
        target = {'target': labels}

        if 'edges' in batch:
            bd_gt = batch['edges']
            target['bd_gt'] = bd_gt.to(self.devices)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            train_model = set_training_targets(train_model, target)
            out = train_model(images)
            loss_factory.calc(out, target, phase='train')

        loss_factory.backward(self.grad_scaler)
        if self.max_norm:
            # Unscales the gradients of optimizer's assigned parameters in-place
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), self.max_norm)
        # optimizer's gradients are already unscaled, so scaler.step doesn't unscale them,
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

        pred = self.postprocessor(out, original_shape=images[0].shape)

        labels = labels.detach().cpu().numpy() # Change it to numpy before compute metric
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [metric_factory.update(g_pred, g_labels, phase='train') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            metric_factory.update(pred, labels, phase='train')

    def valid_step(self, eval_model, batch, loss_factory, metric_factory):
        eval_model.eval()
        indices = batch['indices']
        images = batch['pixel_values'].to(self.devices)
        labels = batch['labels'].long().to(self.devices)
        target = {'target': labels}

        if 'edges' in batch:
            bd_gt = batch['edges']
            target['bd_gt'] = bd_gt.to(self.devices)

        out = eval_model(images)
        loss_factory.calc(out, target, phase='valid')

        pred = self.postprocessor(out, original_shape=images[0].shape)

        indices = indices.numpy()
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
                [metric_factory.update(g_pred, g_labels, phase='valid') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            metric_factory.update(pred, labels, phase='valid')

        logs = {
            'images': images.detach().cpu().numpy(),
            'target': labels,
            'pred': pred
        }
        if 'edges' in batch:
            logs.update({
                'bd_gt': bd_gt.detach().cpu().numpy()
            })
        return dict(logs.items())

    def test_step(self, test_model, batch):
        test_model.eval()
        indices = batch['indices']
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = test_model(images)

        pred = self.postprocessor(out, original_shape=images[0].shape)

        indices = indices.numpy()
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]

            # Remove dummy samples, they only come in distributed environment
            pred = pred[indices != -1]
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_pred = sum(gathered_pred, [])
                pred = gathered_pred

        if self.single_gpu_or_rank_zero:
            results = {
                'images': images.detach().cpu().numpy(),
                'pred': pred
            }
            return results

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid'], metric_factory):
        pass

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
        class_keys = class_map.keys()
        for idx in range(len(result['pred'])):
            image = result['images'][idx:idx+1]
            height, width = image.shape[-2:]
            preds = []
            for class_idx in class_keys:
                binary_mask = np.where(result['pred'][idx] == class_idx, 1, 0)
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                                       mode=cv2.RETR_EXTERNAL,
                                                       method=cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                name = class_map[class_idx]
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        segmentation.append(contour)
                if len(segmentation) == 0:
                    continue
                preds.append(
                    {
                        "class": class_idx,
                        "name": name,
                        "polygon": segmentation
                    }
                )
            return_preds.append(
                {
                    "segmentation": preds,
                    "shape": {
                        "width": width,
                        "height": height
                    }
                }
            )
        return return_preds
