from typing import Dict, Type

import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .custom import SGD, RMSprop, Adam

OPTIMIZER_DICT: Dict[str, Type[Optimizer]] = {
    'adamw': optim.AdamW,
    'adam': Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'rmsprop': RMSprop,
    'adamax': optim.Adamax,
    'sgd': SGD,
}
