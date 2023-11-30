from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import boxes as box_ops

from ...models.heads.detection.experimental.detection._utils import Matcher


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
        self.cls_loss = RetinaNetClassificationLoss()
        self.reg_loss = RetinaNetRegressionLoss()
    
    def compute_loss(self, out, target, matched_idxs):
        cls_loss = self.classification_head.compute_loss(out, target, matched_idxs)
        box_loss = self.regression_head.compute_loss(out, target, matched_idxs)

        # TODO: return as dict
        return cls_loss + box_loss

    def forward(self, out: Dict, target: torch.Tensor) -> torch.Tensor:
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
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, target, out, matched_idxs):
        pass


class RetinaNetRegressionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, target, out, matched_idxs):
        pass