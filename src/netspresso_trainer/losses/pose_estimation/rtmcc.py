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

"""
Based on the RTMPose implementation of mmpose.
https://github.com/open-mmlab/mmpose
"""
from itertools import product
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection._utils import BoxCoder, Matcher
from torchvision.ops import boxes as box_ops

from ..common import SigmoidFocalLoss


class RTMCCLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # TODO: Get from config
        self.beta = 10.
        self.label_softmax = True
        self.label_beta = 10.
        self.use_target_weight = True
        self.mask = None
        self.mask_weight = 1.
        self.simcc_split_ratio = 2.
        self.input_size = (256, 256)

        self.sigma = (5.66, 5.66)
        self.normalize = False

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def _map_coordinates(
        self,
        keypoints,
        keypoints_visible,
    ):
        """Mapping keypoint coordinates into SimCC space."""

        keypoints_split = keypoints.clone()
        keypoints_split = torch.round(keypoints_split * self.simcc_split_ratio)
        keypoints_split = keypoints_split.to(torch.int64)
        keypoint_weights = keypoints_visible.clone()

        return keypoints_split, keypoint_weights

    def gaussian_smoothng(self, keypoints):
        keypoints_xy = keypoints[..., :2]
        keypoints_visible = keypoints[..., 2]

        N, K, _ = keypoints_xy.shape
        # TODO: Get from config
        w, h = self.input_size
        W = round(w * self.simcc_split_ratio)
        H = round(h * self.simcc_split_ratio)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints_xy, keypoints_visible)

        target_x = torch.zeros((N, K, W), dtype=torch.float32, device=keypoints.device)
        target_y = torch.zeros((N, K, H), dtype=torch.float32, device=keypoints.device)

        # 3-sigma rule
        radius = torch.tensor(self.sigma, device=keypoints.device) * 3

        # xy grid
        x = torch.arange(0, W, 1, dtype=torch.float32, device=keypoints.device)
        y = torch.arange(0, H, 1, dtype=torch.float32, device=keypoints.device)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            target_x[n, k] = torch.exp(-((x - mu_x)**2) / (2 * self.sigma[0]**2))
            target_y[n, k] = torch.exp(-((y - mu_y)**2) / (2 * self.sigma[1]**2))

        if self.normalize:
            norm_value = self.sigma * torch.sqrt(torch.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]

        return target_x, target_y, keypoint_weights

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        pred = out['pred']
        keypoints = target['keypoints']

        N, K, _ = pred.shape
        loss = 0

        coord_split = pred.shape[-1] // 2
        pred_x = pred[..., :coord_split]
        pred_y = pred[..., coord_split:]

        gt_x, gt_y, target_weights = self.gaussian_smoothng(keypoints)

        weight = target_weights.reshape(-1) if self.use_target_weight else 1.0

        for p, t in zip([pred_x, pred_y], [gt_x, gt_y]):
            p = p.reshape(-1, p.size(-1))
            t = t.reshape(-1, t.size(-1))

            t_loss = self.criterion(p, t).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            loss = loss + t_loss.sum()

        return loss / K
