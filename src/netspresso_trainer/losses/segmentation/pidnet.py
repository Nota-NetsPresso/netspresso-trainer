from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PIDNetCrossEntropy', 'BoundaryLoss']

IGNORE_INDEX_NONE_VALUE = -100

NUM_OUTPUTS = 2
BALANCE_WEIGHTS = [0.4, 1.0]

OHEMTHRES = 0.9
OHEMKEEP = 131072


class PIDNetCrossEntropy(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX_NONE_VALUE, weight=None):
        super(PIDNetCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index
        )
        self.boundary_aware = False

    def _forward(self, out: torch.Tensor, target: torch.Tensor):

        return self.loss_fn(out, target)

    def forward(self, out: Dict, target: Dict):
        target = target['target']

        if self.boundary_aware:
            pred, extra_d = out['pred'], out['extra_d']
            filler = torch.ones_like(target) * self.ignore_index
            bd_label = torch.where(torch.sigmoid(extra_d[:, 0, :, :]) > 0.8, target, filler)
            return self._forward(pred, bd_label)

        pred, extra_p = out['pred'], out['extra_p']
        score = [extra_p, pred]
        return sum([w * self._forward(x, target) for (w, x) in zip(BALANCE_WEIGHTS, score)])

class PIDNetBoundaryAwareCrossEntropy(PIDNetCrossEntropy):
    def __init__(self, ignore_index=IGNORE_INDEX_NONE_VALUE, weight=None):
        super().__init__(ignore_index, weight)
        self.boundary_aware = True

# class OhemCrossEntropy(nn.Module):
#     def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
#         super(OhemCrossEntropy, self).__init__()
#         self.thresh = thres
#         self.min_kept = max(1, min_kept)
#         self.ignore_label = ignore_label
#         self.criterion = nn.CrossEntropyLoss(
#             weight=weight,
#             ignore_index=ignore_label,
#             reduction='none'
#         )

#     def _ce_forward(self, score, target):

#         loss = self.criterion(score, target)

#         return loss

#     def _ohem_forward(self, score, target, **kwargs):

#         pred = F.softmax(score, dim=1)
#         pixel_losses = self.criterion(score, target).contiguous().view(-1)
#         mask = target.contiguous().view(-1) != self.ignore_label

#         tmp_target = target.clone()
#         tmp_target[tmp_target == self.ignore_label] = 0
#         pred = pred.gather(1, tmp_target.unsqueeze(1))
#         pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
#         min_value = pred[min(self.min_kept, pred.numel() - 1)]
#         threshold = max(min_value, self.thresh)

#         pixel_losses = pixel_losses[mask][ind]
#         pixel_losses = pixel_losses[pred < threshold]
#         return pixel_losses.mean()

#     def forward(self, score, target):

#         if not (isinstance(score, list) or isinstance(score, tuple)):
#             score = [score]

#         balance_weights = BALANCE_WEIGHTS
#         sb_weights = SB_WEIGHTS
#         if len(balance_weights) == len(score):
#             functions = [self._ce_forward] * \
#                 (len(balance_weights) - 1) + [self._ohem_forward]
#             return sum([
#                 w * func(x, target)
#                 for (w, x, func) in zip(balance_weights, score, functions)
#             ])

#         elif len(score) == 1:
#             return sb_weights * self._ohem_forward(score[0], target)

#         else:
#             raise ValueError("lengths of prediction and target are not identical!")


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    @staticmethod
    def weighted_bce(bd_pre, target):
        n, c, h, w = bd_pre.size()
        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = target.view(1, -1)

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

        return loss

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        bd_gt = target['bd_gt']
        extra_d = out['extra_d']
        return self.weighted_bce(extra_d, bd_gt)


class PIDNetLoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX_NONE_VALUE, weight=None):
        super().__init__()

        self.cross_entropy_loss = PIDNetCrossEntropy(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss()
        self.cross_entropy_with_boundary = PIDNetBoundaryAwareCrossEntropy(ignore_index=ignore_index)

    def forward(self, out: Dict, target: Dict):

        cross_entropy_loss = self.cross_entropy_loss(out, target)
        boundary_loss = self.boundary_loss(out, target)
        cross_entropy_loss_with_boundary = self.cross_entropy_with_boundary(out, target)

        # TODO: return as dict
        loss = cross_entropy_loss + 20 * boundary_loss + cross_entropy_loss_with_boundary
        return loss
