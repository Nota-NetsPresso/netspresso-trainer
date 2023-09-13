from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index, **kwargs) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, **kwargs)

    def forward(self, out: Dict, target: torch.Tensor) -> torch.Tensor:
        pred = out['pred']
        loss = self.loss_fn(pred, target)
        return loss