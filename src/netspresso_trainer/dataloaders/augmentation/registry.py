from typing import Dict, Callable

from .custom import RandomHorizontalFlip, RandomResizedCrop

TRANSFORM_DICT: Dict[str, Callable] = {
    'randomresizedcrop': RandomResizedCrop,
    'randomhorizontalflip': RandomHorizontalFlip
}
