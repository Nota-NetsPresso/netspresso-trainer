import math
from typing import List, Tuple, Union

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....op.custom import ConvLayer
from ....utils import FXTensorListType, ModelOutput


class RTMCC(nn.Module):
    def __init__(
            self,
            num_classes: int,
            intermediate_features_dim: List[int],
            params: DictConfig,
    ):
        super().__init__()

    def forward(self, encoder_hidden_states: FXTensorListType):
        return ModelOutput(pred=None)


def rtmcc(num_classes, intermediate_features_dim, conf_model_head, **kwargs) -> RTMCC:
    return RTMCC(num_classes,
                 intermediate_features_dim,
                 params=conf_model_head.params)
