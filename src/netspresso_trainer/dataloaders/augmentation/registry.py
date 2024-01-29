from typing import Callable, Dict

from .custom import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    Mixing,
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
from .custom.mosaic import MosaicDetection


TRANSFORM_DICT: Dict[str, Callable] = {
    'centercrop': CenterCrop,
    'colorjitter': ColorJitter,
    'pad': Pad,
    'randomcrop': RandomCrop,
    'randomresizedcrop': RandomResizedCrop,
    'randomhorizontalflip': RandomHorizontalFlip,
    'randomverticalflip': RandomVerticalFlip,
    'randomerasing': RandomErasing,
    'resize': Resize,
    'mixing': Mixing,
    'mixup': RandomMixup,
    'mosaicdetection': MosaicDetection,
    'cutmix': RandomCutmix,
    'trivialaugmentwide': TrivialAugmentWide,
    'autoaugment': AutoAugment,
}
