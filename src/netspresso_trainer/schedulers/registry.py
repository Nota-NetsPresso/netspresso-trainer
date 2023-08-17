from typing import Dict, Type

import torch
from torch.optim.lr_scheduler import _LRScheduler

from .cosine_warm_restart import CosineAnnealingWarmRestartsWithCustomWarmUp
from .cosine_lr import CosineAnnealingLRWithCustomWarmUp
from .poly_lr import PolynomialLRWithWarmUp
from .step_lr import StepLR

SCHEDULER_DICT: Dict[str, Type[_LRScheduler]] = {
    'cosine': CosineAnnealingWarmRestartsWithCustomWarmUp,
    'cosine_no_sgdr': CosineAnnealingLRWithCustomWarmUp,
    'poly': PolynomialLRWithWarmUp,
    'step': StepLR
}