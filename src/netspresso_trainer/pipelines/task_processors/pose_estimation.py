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


class PoseEstimationProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(PoseEstimationProcessor, self).__init__(conf, postprocessor, devices, **kwargs)

    def train_step(self, train_model, batch, optimizer, loss_factory, metric_factory):
        train_model.train()
        images, keypoints = batch['pixel_values'], batch['keypoints']
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            out = train_model(images, target=target)
            loss_factory.calc(out, target, phase='train')

        loss_factory.backward(self.grad_scaler)
        if self.max_norm:
            # Unscales the gradients of optimizer's assigned parameters in-place
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), self.max_norm)
        # optimizer's gradients are already unscaled, so scaler.step doesn't unscale them,
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

        pred = self.postprocessor(out)

        keypoints = keypoints.detach().cpu().numpy()
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(keypoints, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                pred = np.concatenate(gathered_pred, axis=0)
                keypoints = np.concatenate(gathered_labels, axis=0)

        if self.single_gpu_or_rank_zero:
            logs = {
                'target': keypoints,
                'pred': pred
            }
            return dict(logs.items())

    def valid_step(self, eval_model, batch, loss_factory, metric_factory):
        eval_model.eval()
        indices, images, keypoints = batch['indices'], batch['pixel_values'], batch['keypoints']
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        out = eval_model(images)
        loss_factory.calc(out, target, phase='valid')

        pred = self.postprocessor(out)

        indices = indices.numpy()
        keypoints = keypoints.detach().cpu().numpy()
        if self.conf.distributed:
            pred = pred[indices != -1]
            keypoints = keypoints[indices != -1]

            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(keypoints, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                pred = np.concatenate(gathered_pred, axis=0)
                keypoints = np.concatenate(gathered_labels, axis=0)

        if self.single_gpu_or_rank_zero:
            logs = {
                'images': images.detach().cpu().numpy(),
                'target': keypoints,
                'pred': pred
            }
            return dict(logs.items())

    def test_step(self, test_model, batch):
        test_model.eval()
        indices, images = batch['indices'], batch['pixel_values']
        images = images.to(self.devices)

        out = test_model(images)

        pred = self.postprocessor(out)

        indices = indices.numpy()
        if self.conf.distributed:
            pred = pred[indices != -1]

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
            pred = np.concatenate([output['pred']for output in outputs], axis=0)
            keypoints = np.concatenate([output['target']for output in outputs], axis=0)
            metric_factory.update(pred, keypoints, phase=phase)
