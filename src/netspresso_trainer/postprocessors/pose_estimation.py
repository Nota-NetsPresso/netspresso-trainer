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
Based on the SimCCLabel decode implementation of mmpose.
https://github.com/open-mmlab/mmpose
"""
from typing import Optional

import torch

from ..models.utils import ModelOutput


class PoseEstimationPostprocessor():
    def __init__(self, conf_model):
        params = conf_model.postprocessor.params
        self.simcc_split_ratio = params.simcc_split_ratio

    def get_simcc_maximum(self, simcc_x, simcc_y, apply_softmax: bool = False):
        assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (f'Invalid shape {simcc_x.shape}')
        assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (f'Invalid shape {simcc_y.shape}')
        assert simcc_x.ndim == simcc_y.ndim, (f'{simcc_x.shape} != {simcc_y.shape}')

        N, K, _ = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)

        # if apply_softmax:
        #     simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        #     simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        #     ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        #     simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        #     simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

        x_locs = torch.argmax(simcc_x, axis=1)
        y_locs = torch.argmax(simcc_y, axis=1)
        locs = torch.stack((x_locs, y_locs), axis=-1).to(torch.float32)
        max_val_x = torch.amax(simcc_x, axis=1)
        max_val_y = torch.amax(simcc_y, axis=1)

        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1

        if N:
            locs = locs.reshape(N, K, 2)
            vals = vals.reshape(N, K)

        return locs, vals

    def __call__(self, outputs: ModelOutput):
        pred = outputs['pred']
        coord_split = pred.shape[-1] // 2

        simcc_x = pred[..., :coord_split]
        simcc_y = pred[..., coord_split:]

        keypoints, scores = self.get_simcc_maximum(simcc_x, simcc_y)

        # TODO: use_dark option
        # if self.use_dark:
        #     x_blur = int((self.sigma[0] * 20 - 7) // 3)
        #     y_blur = int((self.sigma[1] * 20 - 7) // 3)
        #     x_blur -= int((x_blur % 2) == 0)
        #     y_blur -= int((y_blur % 2) == 0)
        #     keypoints[:, :, 0] = refine_simcc_dark(keypoints[:, :, 0], simcc_x,
        #                                         x_blur)
        #     keypoints[:, :, 1] = refine_simcc_dark(keypoints[:, :, 1], simcc_y,
        #                                         y_blur)

        keypoints /= self.simcc_split_ratio

        # TODO: decode_visibility option
        # if self.decode_visibility:
        #     _, visibility = get_simcc_maximum(
        #         simcc_x * self.decode_beta * self.sigma[0],
        #         simcc_y * self.decode_beta * self.sigma[1],
        #         apply_softmax=True)
        #     return keypoints, (scores, visibility)
        # else:
        #     return keypoints, scores
        return keypoints.detach().cpu().numpy() #, scores.detach().cpu().numpy()
