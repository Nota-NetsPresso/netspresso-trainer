from typing import Dict, Type

import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

OPTIMIZER_DICT: Dict[str, Type[Optimizer]] = {
    'adamw': optim.AdamW,
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'rmsprop': optim.RMSprop,
    'adamax': optim.Adamax,
    'sgd': optim.SGD,
    'nesterov': optim.SGD,
    'momentum': optim.SGD,
}