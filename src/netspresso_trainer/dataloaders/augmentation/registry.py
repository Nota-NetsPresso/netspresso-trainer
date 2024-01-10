from typing import Callable, Dict

from .custom import (
    AutoAugment,
    ColorJitter,
    Pad,
    RandomCrop,
    RandomCutmix,
    RandomErasing,
    RandomHorizontalFlip,
    RandomMixup,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    TrivialAugmentWide,
)

TRANSFORM_DICT: Dict[str, Callable] = {
    'colorjitter': ColorJitter,
    'pad': Pad,
    'randomcrop': RandomCrop,
    'randomresizedcrop': RandomResizedCrop,
    'randomhorizontalflip': RandomHorizontalFlip,
    'randomverticalflip': RandomVerticalFlip,
    'randomerasing': RandomErasing,
    'resize': Resize,
    'mixup': RandomMixup,
    'cutmix': RandomCutmix,
    'trivialaugmentwide': TrivialAugmentWide,
    'autoaugment': AutoAugment,
}
