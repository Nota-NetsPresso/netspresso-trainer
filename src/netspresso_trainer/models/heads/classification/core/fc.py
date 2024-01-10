from typing import Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from torch.fx.proxy import Proxy

from ....utils import ModelOutput


class FC(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, params: DictConfig) -> None:
        super(FC, self).__init__()
        hidden_size = params.intermediate_channels
        num_layers = params.num_layers

        assert num_layers >= 1, "num_hidden_layers must be integer larger than 0"

        prev_size = feature_dim
        classifier = []
        for _ in range(num_layers - 1):
            classifier.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        classifier.append(nn.Linear(prev_size, num_classes))
        self.classifier = nn.Sequential(*classifier)
        
    def forward(self, x: Union[Tensor, Proxy]) -> ModelOutput:
        x = self.classifier(x)
        return ModelOutput(pred=x)
    
def fc(feature_dim, num_classes, conf_model_head, **kwargs) -> FC:
    return FC(feature_dim=feature_dim, num_classes=num_classes, params=conf_model_head.params)