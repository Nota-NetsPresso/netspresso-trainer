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
