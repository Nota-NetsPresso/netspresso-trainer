from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection._utils import BoxCoder, Matcher
from torchvision.ops import boxes as box_ops

from ..common import SigmoidFocalLoss


class RTMCCLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, out: Dict, target: Dict) -> torch.Tensor:
        return None
