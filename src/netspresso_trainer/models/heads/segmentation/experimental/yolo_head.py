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
