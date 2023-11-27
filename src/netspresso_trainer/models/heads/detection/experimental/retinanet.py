from typing import List

from omegaconf import DictConfig
import torch.nn as nn


class RetinaNetHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super().__init__()
        pass

    def forward(self, xin):
        pass


def retinanet_head(num_classes, intermediate_features_dim, conf_model_head, **kwargs) -> RetinaNetHead:
    return RetinaNetHead(num_classes=num_classes,
                     intermediate_features_dim=intermediate_features_dim,
                     params=conf_model_head.params)
