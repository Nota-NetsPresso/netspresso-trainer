from typing import Literal

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
            out = train_model(images)
            loss_factory.calc(out, target, phase='train')

        loss_factory.backward(self.grad_scaler)
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
