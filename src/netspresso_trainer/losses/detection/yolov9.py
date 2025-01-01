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
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from netspresso_trainer.models.op.custom import Anchor2Vec
from netspresso_trainer.utils.bbox_utils import generate_anchors

from .yolox import IOUloss

"""
    This implementation is based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/tools/loss_functions.py.
"""


def calculate_iou(bbox1, bbox2, metrics="iou") -> Tensor:
    #TODO: It should be consolidated into bbox_utils
    metrics = metrics.lower()
    EPS = 1e-7
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + EPS)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)



class BoxMatcher:
    # TODO: It should be consolidated into bbox_utils
    def __init__(self,
                 class_num: int,
                 anchors: Tensor,

                 ) -> None:
        self.class_num = class_num
        self.anchors = anchors
        self.iou = "ciou"
        self.topk = 10 # TODO: It should be parameterized.
        self.factor = {"iou": 6.0, "cls": 0.5} # TODO: It should be parameterized.

    def get_valid_matrix(self, target_bbox: Tensor):
        """
        Get a boolean mask that indicates whether each target bounding box overlaps with each anchor.

        Args:
            target_bbox [batch x targets x 4]: The bounding box of each targets.
        Returns:
            [batch x targets x anchors]: A boolean tensor indicates if target bounding box overlaps with anchors.
        """
        Xmin, Ymin, Xmax, Ymax = target_bbox[:, :, None].unbind(3)
        anchors = self.anchors[None, None]  # add a axis at first, second dimension
        anchors_x, anchors_y = anchors.unbind(dim=3)
        target_in_x = (Xmin < anchors_x) & (anchors_x < Xmax)
        target_in_y = (Ymin < anchors_y) & (anchors_y < Ymax)
        target_on_anchor = target_in_x & target_in_y
        return target_on_anchor

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """
        Get the (predicted class' probabilities) corresponding to the target classes across all anchors

        Args:
            predict_cls [batch x anchors x class]: The predicted probabilities for each class across each anchor.
            target_cls [batch x targets]: The class index for each target.

        Returns:
            [batch x targets x anchors]: The probabilities from `pred_cls` corresponding to the class indices specified in `target_cls`.
        """
        predict_cls = predict_cls.transpose(1, 2)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox, target_bbox) -> Tensor:
        """
        Get the IoU between each target bounding box and each predicted bounding box.

        Args:
            predict_bbox [batch x predicts x 4]: Bounding box with [x1, y1, x2, y2].
            target_bbox [batch x targets x 4]: Bounding box with [x1, y1, x2, y2].
        Returns:
            [batch x targets x predicts]: The IoU scores between each target and predicted.
        """
        return calculate_iou(target_bbox, predict_bbox, self.iou).clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, topk: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Filter the top-k suitability of targets for each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            topk (int, optional): Number of top scores to retain per anchor.

        Returns:
            topk_targets [batch x targets x anchors]: Only leave the topk targets for each anchor
            topk_masks [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.
        """
        values, indices = target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_masks = topk_targets > 0
        return topk_targets, topk_masks

    def filter_duplicates(self, target_matrix: Tensor, topk_mask: Tensor):
        """
        Filter the maximum suitability target index of each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors

        Returns:
            unique_indices [batch x anchors x 1]: The index of the best targets for each anchors
        """
        duplicates = (topk_mask.sum(1, keepdim=True) > 1).repeat([1, topk_mask.size(1), 1])
        max_idx = F.one_hot(target_matrix.argmax(1), topk_mask.size(1)).permute(0, 2, 1)
        topk_mask = torch.where(duplicates, max_idx, topk_mask)
        unique_indices = topk_mask.argmax(dim=1)
        return unique_indices[..., None], topk_mask.sum(1), topk_mask

    def __call__(self, target: Tensor, predict: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Matches each target to the most suitable anchor.
        1. For each anchor prediction, find the highest suitability targets.
        2. Match target to the best anchor.
        3. Noramlize the class probilities of targets.

        Args:
            target: The ground truth class and bounding box information
                as tensor of size [batch x targets x 5].
            predict: Tuple of predicted class and bounding box tensors.
                Class tensor is of size [batch x anchors x class]
                Bounding box tensor is of size [batch x anchors x 4].

        Returns:
            anchor_matched_targets: Tensor of size [batch x anchors x (class + 4)].
                A tensor assigning each target/gt to the best fitting anchor.
                The class probabilities are normalized.
            valid_mask: Bool tensor of shape [batch x anchors].
                True if a anchor has a target/gt assigned to it.
        """
        predict_cls, predict_bbox = predict

        # return if target has no gt information.
        n_targets = target.shape[1]
        if n_targets == 0:
            device = predict_bbox.device
            align_cls = torch.zeros_like(predict_cls, device=device)
            align_bbox = torch.zeros_like(predict_bbox, device=device)
            valid_mask = torch.zeros(predict_cls.shape[:2], dtype=bool, device=device)
            anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
            return anchor_matched_targets, valid_mask

        target_cls, target_bbox = target.split([1, 4], dim=-1)  # B x N x (C B) -> B x N x C, B x N x B
        target_cls = target_cls.long().clamp(0)

        # get valid matrix (each gt appear in which anchor grid)
        grid_mask = self.get_valid_matrix(target_bbox)

        # get iou matrix (iou with each gt bbox and each predict anchor)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)

        # get cls matrix (cls prob with each gt class and each predict class)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        target_matrix = grid_mask * (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"])

        # choose topk
        topk_targets, topk_mask = self.filter_topk(target_matrix, topk=self.topk)

        # delete one anchor pred assign to mutliple gts
        unique_indices, valid_mask, topk_mask = self.filter_duplicates(iou_mat, topk_mask)

        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls = torch.gather(target_cls, 1, unique_indices).squeeze(-1)
        align_cls = F.one_hot(align_cls, self.class_num)

        # normalize class ditribution
        iou_mat *= topk_mask
        target_matrix *= topk_mask
        max_target = target_matrix.amax(dim=-1, keepdim=True)
        max_iou = iou_mat.amax(dim=-1, keepdim=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]
        anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
        return anchor_matched_targets, valid_mask.bool()




class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 using pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, anchor_grid, scaler, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (anchor_grid / scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOv9Loss(nn.Module):
    def __init__(self, l1_activate_epoch, cur_epoch=None, **kwargs):
        super().__init__()
        self.iou_loss = IOUloss(reduction="none", loss_type="ciou")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.anchor_grid = None
        self.scaler = None
        self.cls = BCELoss()
        self.iou = BoxLoss()
        self.reg_max = kwargs.get("reg_max", 16)
        self.anc2vec = Anchor2Vec(self.reg_max)
        self.aux_rate = 0.25 # TODO: should be controlled by config

    def get_output(self, output, anchor_grid, scaler):
        pred_bbox_reg, pred_bbox_anchor, pred_class_logits = [], [], []
        for layer_output in output:
            reg, class_logits = torch.split(layer_output, [layer_output.shape[1] - self.num_classes, self.num_classes], dim=1)
            bbox_anchor, bbox_reg = self.anc2vec(reg)
            b, c, _, _ = bbox_reg.shape
            reg = bbox_reg.view(b, c, -1).permute(0, 2, 1)
            pred_bbox_reg.append(reg)

            b, a, r, _, _ = bbox_anchor.shape
            anchor = bbox_anchor.view(b, a, r, -1).permute(0, 3, 2, 1)
            pred_bbox_anchor.append(anchor)

            b, c, _, _ = class_logits.shape
            logits = class_logits.view(b, c, -1).permute(0, 2, 1)
            pred_class_logits.append(logits)

        pred_bbox_reg = torch.concat(pred_bbox_reg, dim=1)
        pred_bbox_anchor = torch.concat(pred_bbox_anchor, dim=1)
        pred_class_logits = torch.concat(pred_class_logits, dim=1)

        pred_xyxy = pred_bbox_reg * scaler.view(1, -1, 1)
        lt, rb = pred_xyxy.chunk(2, dim=-1)
        pred_bbox_reg = torch.cat([anchor_grid - lt, anchor_grid + rb], dim=-1)
        return pred_class_logits, pred_bbox_anchor, pred_bbox_reg

    def forward(self, out: Union[List, Dict], target: Dict) -> Tensor:
        # aux_out = out['pred']['aux_outputs'] if isinstance(out['pred'], Dict) else None
        out = out['pred']['outputs'] if isinstance(out['pred'], Dict) else out['pred']
        device = out[0][0].device
        self.num_classes = target['num_classes']
        img_size = target['img_size']
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        target = target['gt']

        strides = []
        for _k, o in enumerate(out):
            stride_this_level = img_size[-1] // o.size(-1)
            strides.append(stride_this_level)
        anchor_grid, scaler = generate_anchors(img_size, strides)
        anchor_grid = anchor_grid.to(device)
        scaler = scaler.to(device)

        dfl = DFLoss(anchor_grid, scaler, self.reg_max)
        preds_cls, preds_anc, preds_box = self.get_output(out, anchor_grid=anchor_grid, scaler=scaler)
        # if aux_out:
        #     aux_preds_cls, aux_preds_anc, aux_preds_box = self.get_output(aux_out, anchor_grid=anchor_grid, scaler=scaler)
        # else:
        #     aux_preds_cls, aux_preds_anc, aux_preds_box = None, None, None

        matcher = BoxMatcher(self.num_classes, anchor_grid)
        max_samples = max([len(t['labels']) for t in target])
        labels = torch.zeros(len(target), max_samples, 5).to(device)
        labels[:, :, 0] = -1
        for batch_idx, t in enumerate(target):
            batch_boxes = t['boxes']
            batch_labels = t['labels']
            sample_num = len(batch_labels)
            labels[batch_idx, :sample_num, 1:5] = batch_boxes
            labels[batch_idx, :sample_num, 0] = batch_labels

        align_targets, valid_masks = matcher(labels, (preds_cls.detach(), preds_box.detach()))
        targets_cls, targets_bbox = self.separate_anchor(align_targets, scaler)
        preds_box = preds_box / scaler[None, :, None]
        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]
        # if aux_out:
        #     aux_align_targets, aux_valid_masks = matcher(labels, (aux_preds_cls.detach(), aux_preds_box.detach()))
        #     aux_preds_box = aux_preds_box / scaler[None, :, None]
        #     aux_targets_cls, aux_targets_bbox = self.separate_anchor(aux_align_targets, scaler)
        #     aux_cls_norm = max(aux_targets_cls.sum(), 1)
        #     aux_box_norm = aux_targets_cls.sum(-1)[aux_valid_masks]
        #     ## -- CLS -- ##
        #     aux_loss_cls = self.cls(aux_preds_cls, aux_targets_cls, aux_cls_norm)
        #     ## -- IOU -- ##
        #     aux_loss_iou = self.iou(aux_preds_box, aux_targets_bbox, aux_valid_masks, aux_box_norm, aux_cls_norm)
        #     ## -- DFL -- ##
        #     aux_loss_dfl = dfl(aux_preds_anc, aux_targets_bbox, aux_valid_masks, aux_box_norm, aux_cls_norm)


        ## -- CLS -- ##
        loss_cls = self.cls(preds_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(preds_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = dfl(preds_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        # TODO: loss weights should be controlled by config
        # if aux_out:
        #     total_loss = 0.5 * (self.aux_rate * aux_loss_cls + loss_cls) + 1.5 * (self.aux_rate * aux_loss_dfl + loss_dfl) + 7.5 * (self.aux_rate * aux_loss_iou + loss_iou)
        # else:
        total_loss = 0.5 * loss_cls + 1.5 * loss_dfl + 7.5 * loss_iou
        return total_loss

    def separate_anchor(self, anchors, scaler):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.num_classes, 4), dim=-1)
        anchors_box = anchors_box / scaler[None, :, None]
        return anchors_cls, anchors_box
