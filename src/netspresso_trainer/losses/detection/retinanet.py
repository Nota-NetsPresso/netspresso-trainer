from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection._utils import BoxCoder, Matcher
from torchvision.ops import boxes as box_ops

from ..common import SigmoidFocalLoss


class RetinaNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.4
        self.proposal_matcher = Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
        )
        self.cls_loss = RetinaNetClassificationLoss(self.proposal_matcher.BETWEEN_THRESHOLDS)
        self.reg_loss = RetinaNetRegressionLoss()

    def compute_loss(self, out, target, matched_idxs):
        cls_loss = self.cls_loss(out, target, matched_idxs)
        box_loss = self.reg_loss(out, target, matched_idxs)

        # TODO: return as dict
        return cls_loss + box_loss

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        matched_idxs = []
        anchors = out['anchors']

        for targets_per_image in target['gt']:
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors.size(0),), -1, dtype=torch.int64, device=anchors.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.compute_loss(out, target, matched_idxs)


class RetinaNetClassificationLoss(nn.Module):
    def __init__(self, between_threshold) -> None:
        super().__init__()
        self.between_threshold = between_threshold
        # TODO: Get from config
        alpha = 0.25
        gamma = 2
        self.focal_loss = SigmoidFocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, out, target, matched_idxs):
        losses = []

        cls_logits = out["cls_logits"]
        cls_logits = torch.cat(cls_logits, dim=1)

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(target['gt'], cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.between_threshold

            # compute the classification loss
            losses.append(
                self.focal_loss(
                    {'pred': cls_logits_per_image[valid_idxs_per_image]},
                    {'target': gt_classes_target[valid_idxs_per_image]},
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return sum(losses) / len(target['gt'])


class RetinaNetRegressionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def forward(self, out, target, matched_idxs):
        losses = []

        bbox_regression = out["bbox_regression"]
        bbox_regression = torch.cat(bbox_regression, dim=1)
        anchors = out["anchors"]

        for targets_per_image, bbox_regression_per_image, matched_idxs_per_image in zip(
            target['gt'], bbox_regression, matched_idxs
        ):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors[foreground_idxs_per_image, :]

            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            loss = F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum")

            # compute the loss
            losses.append(loss / max(1, num_foreground))

        return sum(losses) / max(1, len(target['gt']))
