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

import numpy as np
import torch

from .base import BaseTaskProcessor


class ClassificationProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(ClassificationProcessor, self).__init__(conf, postprocessor, devices, **kwargs)

    def train_step(self, train_model, batch, optimizer, loss_factory, metric_factory):
        train_model.train()
        indices, images, labels = batch
        images = images.to(self.devices).to(self.data_type)
        labels = labels.to(self.devices).to(self.data_type)
        target = {'target': labels}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            if isinstance(train_model, torch.nn.parallel.DistributedDataParallel):
                if hasattr(train_model.module.head, "_training_targets"): # for RT-DETR
                    train_model.module.head.set_training_targets(target)
            else:
                if hasattr(train_model.head, "_training_targets"):
                    train_model.head.set_training_targets(target)
            out = train_model(images)
            loss_factory.calc(out, target, phase='train')
        if labels.dim() > 1: # Soft label to label number
            labels = torch.argmax(labels, dim=-1)
        pred, _ = self.postprocessor(out)

        loss_factory.backward(self.grad_scaler)
        if self.max_norm:
            # Unscales the gradients of optimizer's assigned parameters in-place
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), self.max_norm)
        # optimizer's gradients are already unscaled, so scaler.step doesn't unscale them,
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

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
        indices, images, labels = batch
        images = images.to(self.devices)
        labels = labels.to(self.devices)
        target = {'target': labels}

        out = eval_model(images)
        loss_factory.calc(out, target, phase='valid')
        if labels.dim() > 1: # Soft label to label number
            labels = torch.argmax(labels, dim=-1)
        pred, conf_score = self.postprocessor(out)

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

        if self.single_gpu_or_rank_zero:
            results = {
                'images': images.detach().cpu().numpy(),
                'pred': {"label": pred, "conf_score": conf_score}
            }
            return results

    def test_step(self, test_model, batch):
        test_model.eval()
        indices, images, _ = batch
        images = images.to(self.devices)

        out = test_model(images)
        pred, conf_score = self.postprocessor(out, k=1)

        indices = indices.numpy()
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]

            # Remove dummy samples, they only come in distributed environment
            pred = pred[indices != -1]
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_pred = np.concatenate(gathered_pred, axis=0)
                pred = gathered_pred

        if self.single_gpu_or_rank_zero:
            results = {
                'images': images.detach().cpu().numpy(),
                'pred': {"label": pred, "conf_score": conf_score}
            }
            return results

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid'], metric_factory):
        pass

    def _convert_result(self, result, class_map):
        assert "pred" in result and "images" in result
        return_preds = []
        for idx in range(len(result['images'])):
            image = result['images'][idx:idx+1]
            height, width = image.shape[-2:]
            label = result['pred']['label'][idx]
            conf_score = result['pred']['conf_score'][idx]
            return_preds.append(
                {
                    "class": int(label[0]),
                    "name": class_map[int(label[0])],
                    "conf_score": float(conf_score[0]),
                    "shape": {
                        "width": width,
                        "height": height
                    }
                }
            )
        return return_preds

    def get_predictions(self, results, class_map):
        predictions = []
        if isinstance(results, list):
            for minibatch in results:
                predictions.extend(self._convert_result(minibatch, class_map))
        elif isinstance(results, dict):
            predictions.extend(self._convert_result(results, class_map))

        return predictions
