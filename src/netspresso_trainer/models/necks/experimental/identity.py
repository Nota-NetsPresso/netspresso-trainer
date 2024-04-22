from typing import List

from omegaconf import DictConfig
import torch.nn as nn
import torch
import torch.nn.functional as F

from ...utils import BackboneOutput


class Identity(nn.Module):

    def __init__(
        self,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super(Identity, self).__init__()

        self._intermediate_features_dim = (128, 96)

    def forward(self, inputs):

        # a = torch.ones(128, 128, 1, 1).to(device='cuda')
        # b = torch.ones(96, 96, 1 ,1).to(device='cuda')

        a = torch.ones(2, 128, 20, 20).to(device='cuda')
        b = torch.ones(2, 96, 40, 40).to(device='cuda')

        return BackboneOutput(intermediate_features=(a,b))

    


        return BackboneOutput(intermediate_features=inputs)
    
    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim


def identity(intermediate_features_dim, conf_model_neck, **kwargs) -> Identity:
    return Identity(
        intermediate_features_dim=intermediate_features_dim,
        params=conf_model_neck.params,
    )
