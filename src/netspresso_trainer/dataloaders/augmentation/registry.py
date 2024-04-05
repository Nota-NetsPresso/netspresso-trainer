from typing import Callable, Dict

from .custom.image_proc import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    HSVJitter,
    Pad,
    PoseTopDownAffine,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResize,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    TrivialAugmentWide,
)
from .custom.mixing import Mixing
from .custom.mosaic import MosaicDetection

TRANSFORM_DICT: Dict[str, Callable] = {
    'centercrop': CenterCrop,
    'colorjitter': ColorJitter,
    'pad': Pad,
    'randomcrop': RandomCrop,
    'randomresizedcrop': RandomResizedCrop,
    'randomhorizontalflip': RandomHorizontalFlip,
    'randomresize': RandomResize,
    'randomverticalflip': RandomVerticalFlip,
    'randomerasing': RandomErasing,
    'resize': Resize,
    'mixing': Mixing,
    'mosaicdetection': MosaicDetection,
    'trivialaugmentwide': TrivialAugmentWide,
    'autoaugment': AutoAugment,
    'hsvjitter': HSVJitter,
    'posetopdownaffine': PoseTopDownAffine,
}
