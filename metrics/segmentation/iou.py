import numpy as np
import torch

from utils.common import AverageMeter


IGNORE_INDEX_NONE_VALUE = -100


def _intersection_and_union_gpu(output, target, **kwargs):
    assert 'num_classes' in kwargs
    ignore_index = kwargs['ignore_index'] if 'ignore_index' in kwargs else IGNORE_INDEX_NONE_VALUE
    ignore_index = ignore_index if ignore_index is not None else IGNORE_INDEX_NONE_VALUE
    K = kwargs['num_classes']

    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])

    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection

    intersection, union, target, output = \
        area_intersection.cpu().numpy(), area_union.cpu().numpy(), area_target.cpu().numpy(), area_output.cpu().numpy()

    return {
        'intersection': intersection,
        'union': union,
        'target': target,
        'output': output
    }


def iou(output, target, **kwargs):

    B = output.size(0)
    metric_iou = AverageMeter('iou')

    output_seg = torch.max(output, dim=1)[1]  # argmax
    metrics = _intersection_and_union_gpu(output_seg, target, **kwargs)
    metric_iou.update(sum(metrics['intersection']) / (sum(metrics['union']) + 1e-10), n=B)

    return metric_iou.avg
