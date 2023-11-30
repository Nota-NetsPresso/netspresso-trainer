from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RetinaNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(out: Dict, target: torch.Tensor) -> torch.Tensor:
        return None