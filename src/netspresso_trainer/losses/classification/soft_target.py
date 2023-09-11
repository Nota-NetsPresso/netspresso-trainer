from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module): # cutmix/mixup augmentation
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, out: Dict, target: torch.Tensor) -> torch.Tensor:
        pred = out['pred']
        loss = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        return loss.mean()