from typing import Dict, Type
from pathlib import Path

import torch
import torch.nn as nn

PRETRAINED_ROOT = Path("/CHECKPOINT")  # TODO: as an option

SUPPORTING_MODEL_LIST = ["atomixnet_l", "atomixnet_m", "atomixnet_s", "resnet50", "segformer", "pidnet", "mobilevit", "vit", "efficientformer"]
MODEL_PRETRAINED_DICT = {
    "atomixnet_l": PRETRAINED_ROOT / "backbones" / "atomixnet" / "atomixnet_l.pth",
    "atomixnet_m": PRETRAINED_ROOT / "backbones" / "atomixnet" / "atomixnet_m.pth",
    "atomixnet_s": PRETRAINED_ROOT / "backbones" / "atomixnet" / "atomixnet_s.pth",
    "resnet50": PRETRAINED_ROOT / "backbones" / "resnet" / "resnet50.pth",
    "segformer": PRETRAINED_ROOT / "backbones" / "segformer" / "segformer.pth",
    "pidnet": PRETRAINED_ROOT / "full" / "pidnet" / "pidnet_s.pth",
    "mobilevit": PRETRAINED_ROOT / "backbones" / "mobilevit" / "mobilevit_s.pth",
    "vit": PRETRAINED_ROOT / "backbones" / "vit" / "vit_without_qkv.pth",
    "efficientformer": PRETRAINED_ROOT / "backbones" / "efficientformer" / "efficientformer_l1_1000d.pth",
}

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
}