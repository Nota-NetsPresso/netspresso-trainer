from typing import List

import numpy as np
import torch

from ..base import BaseMetric


class PoseEstimationMetric(BaseMetric):
    metric_names: List[str] = ['pck']
    primary_metric: str = 'pck'

    def __init__(self, **kwargs):
        super().__init__()
        # TODO: Get from config
        self.thr = 0.05
        self.input_size = (256, 256)
    
    def _calc_distances(self, preds, gts, mask, norm_factor):
        N, K, _ = preds.shape
        # set mask=0 when norm_factor==0
        _mask = mask.copy()
        _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

        distances = np.full((N, K), -1, dtype=np.float32)
        # handle invalid values
        norm_factor[np.where(norm_factor <= 0)] = 1e6
        distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
        return distances.T
    
    def _distance_acc(self, distances, thr):
        distance_valid = distances != -1
        num_distance_valid = distance_valid.sum()
        if num_distance_valid > 0:
            return (distances[distance_valid] < thr).sum() / num_distance_valid
        return -1

    def keypoint_pck_accuracy(self, pred, gt, mask, thr, norm_factor):
        distances = self._calc_distances(pred, gt, mask, norm_factor)
        acc = np.array([self._distance_acc(d, thr) for d in distances])
        valid_acc = acc[acc >= 0]
        cnt = len(valid_acc)
        avg_acc = valid_acc.mean() if cnt > 0 else 0.0
        # ``acc`` contains accuracy of each keypoint
        return acc, avg_acc, cnt

    def calibrate(self, pred, target, **kwargs):
        result_dict = {k: 0. for k in self.metric_names}

        N, _, _ = pred.shape

        mask = target[..., 2] > 0
        target = target[..., :2]

        # if normalize is None:
        #     normalize = np.tile(np.array([[H, W]]), (N, 1))
        # Normalize by box
        normalize = np.tile(np.array([[self.input_size[0], self.input_size[1]]]), (N, 1))

        acc, avg_acc, cnt = self.keypoint_pck_accuracy(pred, target, mask, self.thr, normalize)
        result_dict['pck'] = avg_acc

        return result_dict
