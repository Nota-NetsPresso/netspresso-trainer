# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch import Tensor
from torch.fx.proxy import Proxy
from typing import List, Union, Optional, Tuple
from ....op.custom import ConvLayer
from ....utils import ModelOutput


class Segmentation(nn.Module):
    """
    A single segmentation head for the YOLO models
    """
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_mask: int,
            act_type: Optional[str] = None
                 ):
        super().__init__()
        mask_neck = max(hidden_channels // 4, num_mask)

        self.mask_convs = nn.Sequential(
            ConvLayer(in_channels, mask_neck, kernel_size=3, act_type=act_type),
            ConvLayer(mask_neck, mask_neck, kernel_size=3, act_type=act_type),
            nn.Conv2d(mask_neck, num_mask, kernel_size=1)
        )
    
    def forward(self, x: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
        x = self.mask_convs(x)
        return x


class YOLOSegmentationHead(nn.Module):
    def __init__(
            self,
            num_classes: int,
            intermediate_features_dim: List[int],
            params: DictConfig,
        ):
        super().__init__()
        self._validate_params(params)
        self.num_classes = num_classes
        self.hidden_dim = int(intermediate_features_dim[0])

        self.heads = self._build_heads(
            intermediate_features_dim,
            params.act_type
        )
    
    def _validate_params(self, params: DictConfig) -> None:
        required_params = ['act_type']
        for param in required_params:
            if not hasattr(params, param):
                raise ValueError(f"Missing required parameter: {param}")
        
    def _build_heads(
            self,
            intermediate_features_dim: List[int],
            act_type: str,
        ):
        heads = nn.ModuleList()
        for feat_dim in intermediate_features_dim[:-1]:
            head = Segmentation(
                int(feat_dim),
                self.hidden_dim,
                self.num_classes,
                act_type=act_type
            )
            heads.append(head)
        heads.append(ConvLayer(
                        int(intermediate_features_dim[-1]), 
                        self.num_classes, 
                        kernel_size=1, 
                        act_type=act_type)
                    )

    def forward(self, x_in: List[Union[Tensor, Proxy]], targets: Optional[Tensor] = None) -> ModelOutput:
        outputs = [head(x) for head, x in zip(self.heads, x_in)]
        return ModelOutput(pred=outputs)


def yolo_segmentation_head(num_classes, intermediate_features_dim, conf_model_head, **kwargs):
    return YOLOSegmentationHead(num_classes,
                                intermediate_features_dim,
                                params=conf_model_head.params)
