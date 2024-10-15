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

from .yolox import IOUloss, YOLOXLoss, xyxy2cxcywh


def xyxy2cxcywhn(bboxes, img_size):
    new_bboxes = bboxes.clone() / img_size
    new_bboxes[:, 2] = new_bboxes[:, 2] - new_bboxes[:, 0]
    new_bboxes[:, 3] = new_bboxes[:, 3] - new_bboxes[:, 1]
    new_bboxes[:, 0] = new_bboxes[:, 0] + new_bboxes[:, 2] * 0.5
    new_bboxes[:, 1] = new_bboxes[:, 1] + new_bboxes[:, 3] * 0.5
    return new_bboxes

def bboxes_iou(bboxes_a, bboxes_b, xyxy=False):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


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
            target[i]['boxes'] = xyxy2cxcywh(target[i]['boxes'])

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
