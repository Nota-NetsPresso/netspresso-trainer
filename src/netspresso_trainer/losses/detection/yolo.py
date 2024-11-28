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

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from netspresso_trainer.utils.bbox_utils import transform_bbox

from .yolox import IOUloss, YOLOXLoss


class YOLOFastestLoss(YOLOXLoss):
    def __init__(self, anchors, l1_activate_epoch=None, cur_epoch=None, **kwargs) -> None:
        super().__init__(l1_activate_epoch, cur_epoch, **kwargs)
        self.iou_loss = IOUloss(reduction="none", loss_type="giou")
        self.anchors = [torch.tensor(anchor, dtype=torch.float).view(-1, 2) for anchor in anchors]
        self.num_anchors = self.anchors[0].size(0)

    def use_l1_update(self):
        return False

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        device = output.device
        batch_size = output.shape[0]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing="ij")
            grid = torch.stack((xv, yv), 2).repeat(self.num_anchors,1,1,1).view(1, self.num_anchors, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        anchors = self.anchors[k].view(1, self.num_anchors, 1, 1, 2).to(device)
        output = output.permute(0, 1, 3, 4, 2)
        output = torch.cat([
            (output[..., :2].sigmoid() + grid) * stride,
            2. * (torch.tanh(output[..., 2:4]/2 -.549306) + 1.) * anchors,
            output[..., 4:]
        ], dim=-1).reshape(
            batch_size, hsize * wsize * self.num_anchors, -1
        )
        return output, grid.view(1, -1, 2)

    def forward(self, out: List, target: Dict) -> torch.Tensor:
        self.use_l1 = self.use_l1_update()

        out = out['pred']
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        self.grids = [torch.zeros(1)] * len(out)
        self.num_classes = target['num_classes']
        img_size = target['img_size']

        target = target['gt']

        out_for_loss = []
        for k, o in enumerate(out):
            stride_this_level = img_size // o.size(-1)

            o, grid = self.get_output_and_grid(
                o, k, stride_this_level, o.type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(o)
            )
            out_for_loss.append(o)

        # YOLOX model learns box cxcywh format directly,
        # but our detection dataloader gives xyxy format.
        for i in range(len(target)):
            target[i]['boxes'] = transform_bbox(target[i]['boxes'], "xyxy -> cxcywh")

        # Ready for l1 loss
        origin_preds = []
        for o in out:
            out_for_l1 = o.view(o.shape[0], self.num_anchors, -1, o.shape[-2], o.shape[-1]).permute(0, 1, 3, 4, 2)
            reg_output = out_for_l1[..., :4]
            batch_size = reg_output.shape[0]
            reg_output = reg_output.reshape(
                batch_size, -1, 4
            )
            origin_preds.append(reg_output.clone())

        total_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.get_losses(
                    None,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    target,
                    torch.cat(out_for_loss, 1),
                    origin_preds,
                    dtype=out[0].dtype,
                )

        # TODO: return as dict
        return total_loss
