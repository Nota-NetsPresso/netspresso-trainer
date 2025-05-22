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

from netspresso_trainer.models.utils import set_training_targets

from ...utils.protocols import ProcessorStepOut
from .base import BaseTaskProcessor


class ClassificationProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(ClassificationProcessor, self).__init__(conf, postprocessor, devices, **kwargs)

    def train_step(self, train_model, batch, optimizer, loss_factory, metric_factory):
        train_model.train()
        images, labels = batch['pixel_values'], batch['labels']
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)

        images = images.to(self.devices).to(self.data_type)
        labels = labels.to(self.devices).to(self.data_type)
        target = {'target': labels}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            train_model = set_training_targets(train_model, target)
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

        return ProcessorStepOut.empty() # Return empty output

    def valid_step(self, eval_model, batch, loss_factory, metric_factory):
        eval_model.eval()
        name = batch['name']
        indices, images, labels = batch['indices'], batch['pixel_values'], batch['labels']
        indices = torch.stack(indices, dim=0)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)

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
            gathered_name = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_conf_score = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            # Remove dummy samples, they only come in distributed environment
            name = np.array(name)[indices != -1].tolist()
            pred = pred[indices != -1]
            conf_score = conf_score[indices != -1]
            labels = labels[indices != -1]
            torch.distributed.gather_object(name, gathered_name if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(conf_score, gathered_conf_score if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [metric_factory.update(g_pred, g_labels, phase='valid') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
                name = sum(gathered_name, [])
                pred = np.concatenate(gathered_pred, axis=0)
                conf_score = np.concatenate(gathered_conf_score, axis=0)
                labels = np.concatenate(gathered_labels, axis=0)
        else:
            metric_factory.update(pred, labels, phase='valid')

        step_out = ProcessorStepOut.empty()
        if self.single_gpu_or_rank_zero:
            step_out['name'] = name
            step_out['pred'] = [
                {'label': label, 'conf_score': conf_score}
                for label, conf_score in zip(list(pred), list(conf_score))
            ]
            step_out['target'] = [
                {'label': np.expand_dims(label, axis=0)} # Add extra dimension to match the pred shape
                for label in labels
            ]

        return step_out

    def test_step(self, test_model, batch):
        test_model.eval()
        name = batch['name']
        indices, images = batch['indices'], batch['pixel_values']
        indices = torch.stack(indices, dim=0)
        images = torch.stack(images, dim=0)

        images = images.to(self.devices)

        out = test_model(images)
        pred, conf_score = self.postprocessor(out, k=1)

        indices = indices.numpy()
        if self.conf.distributed:
            gathered_name = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_conf_score = [None for _ in range(torch.distributed.get_world_size())]

            # Remove dummy samples, they only come in distributed environment
            name = np.array(name)[indices != -1].tolist()
            pred = pred[indices != -1]
            conf_score = conf_score[indices != -1]
            torch.distributed.gather_object(name, gathered_name if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(conf_score, gathered_conf_score if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                name = sum(gathered_name, [])
                pred = np.concatenate(gathered_pred, axis=0)
                conf_score = np.concatenate(gathered_conf_score, axis=0)

        step_out = ProcessorStepOut.empty()
        if self.single_gpu_or_rank_zero:
            step_out['name'] = name
            step_out['pred'] = [
                {'label': label, 'conf_score': conf_score}
                for label, conf_score in zip(list(pred), list(conf_score))
            ]

        return step_out

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid'], metric_factory):
        pass

    def get_predictions(self, results, class_map):
        assert "pred" in results and "name" in results

        predictions = []
        for idx in range(len(results['name'])):
            sample = results['name'][idx]
            label = results['pred'][idx]['label']
            conf_score = results['pred'][idx]['conf_score']
            predictions.append(
                {
                    "sample": sample,
                    "class": int(label[0]),
                    "class_name": class_map[int(label[0])],
                    "conf_score": float(conf_score[0]),
                }
            )

        return predictions
