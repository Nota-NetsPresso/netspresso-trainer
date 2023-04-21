from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, out: Dict, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(out['pred'], target)
        return loss