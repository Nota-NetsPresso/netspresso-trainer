from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[Tensor]=None, size_average=None, ignore_index: int=-100,
                 reduce=None, label_smoothing: float=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                           reduce=reduce, reduction='mean', label_smoothing=label_smoothing)

    def forward(self, out: Dict, target: Dict) -> Dict:
        pred = out['pred']
        target = target['target']
        loss = self.loss_fn(pred, target)
        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out: Dict, target: Dict, reduction='mean'):
        pred = out['pred']
        target = target['target']
        assert pred.shape == target.shape, 'Tensor shapes of prediction and target must be same for SigmoidFocalLoss.'

        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
