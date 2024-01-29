from typing import Callable, Dict

from .custom import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    Pad,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    TrivialAugmentWide,
)
from .custom.mosaic import MosaicDetection
from .custom.mixing import Mixing


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
    'mosaicdetection': MosaicDetection,
    'trivialaugmentwide': TrivialAugmentWide,
    'autoaugment': AutoAugment,
}
