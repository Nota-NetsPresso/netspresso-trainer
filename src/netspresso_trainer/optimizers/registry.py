from typing import Dict, Type

import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .custom import SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad

OPTIMIZER_DICT: Dict[str, Type[Optimizer]] = {
    'adamw': AdamW,
    'adam': Adam,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'rmsprop': RMSprop,
    'adamax': optim.Adamax,
    'sgd': SGD,
}
