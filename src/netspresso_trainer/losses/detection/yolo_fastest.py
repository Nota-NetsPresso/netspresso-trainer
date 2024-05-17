import torch 
import torch.nn as nn
from typing import List, Dict  


class YOLOFastestLoss(nn.Module):
    def __init__(self) -> None:
        super(YOLOFastestLoss, self).__init__()
    
    def forward(self, out: List, target: Dict) -> torch.Tensor:
        pass 




