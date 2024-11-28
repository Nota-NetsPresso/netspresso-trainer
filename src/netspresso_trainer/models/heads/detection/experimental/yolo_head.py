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

"""
Based on the YOLO implementation of WongKinYiu
https://github.com/WongKinYiu/YOLO/blob/main/yolo/model/module.py
"""
import torch
import torch.nn as nn

from typing import Union
from torch import Tensor

def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
    """
        Round up `x` to the biggest-nearest multiple of `div`
    """
    return x + (-x % div)


class Detection(nn.Module):
    """
        A single detection head.
    """
    def __init__(self):
        super().__init__()