import logging
from typing import TypedDict

import thop
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from .environment import get_device


def get_params_and_macs(model: nn.Module, sample_input: torch.Tensor):
    sample_input = sample_input.to(get_device(model))
    # From v0.0.9
    macs, params = _params_and_macs_fvcore(model, sample_input)

    # # Before v0.0.9
    # macs, params = _params_and_macs_thop(model, sample_input)

    return macs, params

def _params_and_macs_fvcore(model: nn.Module, sample_input: torch.Tensor):
    fvcore_logger = logging.getLogger('fvcore')
    fvcore_logger.setLevel(logging.CRITICAL)
    macs = FlopCountAnalysis(model, sample_input).total()
    params = sum(p.numel() for p in model.parameters())
    return macs, params

def _params_and_macs_thop(model: nn.Module, sample_input: torch.Tensor):
    macs, params = thop.profile(model, inputs=(sample_input,), verbose=False)
    return macs, params
