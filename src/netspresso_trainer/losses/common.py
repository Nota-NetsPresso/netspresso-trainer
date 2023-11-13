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

    def forward(self, out: Dict, target: torch.Tensor) -> torch.Tensor:
        pred = out['pred']
        loss = self.loss_fn(pred, target)
        return loss
