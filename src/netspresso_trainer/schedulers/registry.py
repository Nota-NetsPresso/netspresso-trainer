from typing import Dict, Type

from torch.optim.lr_scheduler import _LRScheduler

from .cosine_lr import CosineAnnealingLRWithCustomWarmUp
from .poly_lr import PolynomialLRWithWarmUp

SCHEDULER_DICT: Dict[str, Type[_LRScheduler]] = {
    'cosine': CosineAnnealingLRWithCustomWarmUp,
    'poly': PolynomialLRWithWarmUp
}