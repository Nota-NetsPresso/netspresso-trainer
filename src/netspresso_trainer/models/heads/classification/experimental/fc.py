from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from ....utils import ModelOutput


class FC(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super(FC, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x: Union[Tensor, Proxy]) -> ModelOutput:
        x = self.classifier(x)
        return ModelOutput(pred=x)
    
def fc(feature_dim, num_classes, **kwargs):
    return FC(feature_dim=feature_dim, num_classes=num_classes)