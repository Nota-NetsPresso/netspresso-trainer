from typing import Callable, Dict

from .custom import ColorJitter, Pad, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, RandomVerticalFlip, Resize

TRANSFORM_DICT: Dict[str, Callable] = {
    'colorjitter': ColorJitter,
    'pad': Pad,
    'randomcrop': RandomCrop,
    'randomresizedcrop': RandomResizedCrop,
    'randomhorizontalflip': RandomHorizontalFlip,
    'randomVerticalFlip': RandomVerticalFlip,
    'resize': Resize,
}
