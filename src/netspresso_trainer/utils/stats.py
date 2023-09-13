from typing import TypedDict

import thop
import torch
import torch.nn as nn

from .environment import get_device


def get_params_and_macs(model: nn.Module, sample_input: torch.Tensor):
    sample_input = sample_input.to(get_device(model))
    macs, params = thop.profile(model, inputs=(sample_input,), verbose=False)

    return macs, params
