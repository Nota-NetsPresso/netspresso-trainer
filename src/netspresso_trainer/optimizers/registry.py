from typing import Dict, Type

from torch.optim.optimizer import Optimizer

from .custom import SGD, Adadelta, Adagrad, Adam, Adamax, AdamW, RMSprop

OPTIMIZER_DICT: Dict[str, Type[Optimizer]] = {
    'adamw': AdamW,
    'adam': Adam,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'rmsprop': RMSprop,
    'adamax': Adamax,
    'sgd': SGD,
}
