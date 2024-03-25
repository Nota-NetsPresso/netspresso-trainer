from typing import Literal

import torch

from .base import BaseTaskProcessor


class SegmentationProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices):
        super(SegmentationProcessor, self).__init__(conf, postprocessor, devices)

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
        out = train_model(images)
        loss_factory.calc(out, target, phase='train')

        loss_factory.backward()
        optimizer.step()

        out = {k: v.detach() for k, v in out.items()}
        pred = self.postprocessor(out)

        labels = labels.detach().cpu().numpy() # Change it to numpy before compute metric
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(labels, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                [metric_factory.calc(g_pred, g_labels, phase='train') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            metric_factory.calc(pred, labels, phase='train')

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

        pred = self.postprocessor(out)

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
                [metric_factory.calc(g_pred, g_labels, phase='valid') for g_pred, g_labels in zip(gathered_pred, gathered_labels)]
        else:
            metric_factory.calc(pred, labels, phase='valid')

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

        pred = self.postprocessor(out)

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
