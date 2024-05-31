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

import logging
from typing import TypedDict

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from .environment import get_device


def get_params_and_macs(model: nn.Module, sample_input: torch.Tensor):
    sample_input = sample_input.to(get_device(model))
    # From v0.0.9
    flops, params = _params_and_macs_fvcore(model, sample_input)

    return flops, params

def _params_and_macs_fvcore(model: nn.Module, sample_input: torch.Tensor):
    fvcore_logger = logging.getLogger('fvcore')
    fvcore_logger.setLevel(logging.CRITICAL)
    flops = FlopCountAnalysis(model, sample_input).total()
    params = sum(p.numel() for p in model.parameters())
    return flops, params
