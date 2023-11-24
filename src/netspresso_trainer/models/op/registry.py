from typing import Dict, Type

import torch
import torch.nn as nn

NORM_REGISTRY: Dict[str, Type[nn.Module]] = {
    'batch_norm': nn.BatchNorm2d,
    'instance_norm': nn.InstanceNorm2d,
}

ACTIVATION_REGISTRY: Dict[str, Type[nn.Module]] = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'leaky_relu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'swish': nn.SiLU,
    'hard_swish': nn.Hardswish,
}
