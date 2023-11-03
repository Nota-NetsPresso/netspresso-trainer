from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import MISSING, MissingMandatoryValue

DEFAULT_IMG_SIZE = 256


@dataclass
class Transform:
    name: str = MISSING


@dataclass
class ColorJitter(Transform):
    name: str = 'colorjitter'
    brightness: Optional[float] = 0.25
    contrast: Optional[float] = 0.25
    saturation: Optional[float] = 0.25
    hue: Optional[float] = 0.1
    colorjitter_p: Optional[float] = 0.5


@dataclass
class Pad(Transform):
    name: str = 'pad'
    padding: Union[int, List] = 0


@dataclass
class RandomCrop(Transform):
    name: str = 'randomcrop'
    size: int = DEFAULT_IMG_SIZE
    interpolation: Optional[str] = 'bilinear'


@dataclass
class RandomResizedCrop(Transform):
    name: str = 'randomresizedcrop'
    size: int = DEFAULT_IMG_SIZE
    interpolation: Optional[str] = 'bilinear'


@dataclass
class RandomHorizontalFlip(Transform):
    name: str = 'randomhorizontalflip'
    p: float = 0.5


@dataclass
class RandomVerticalFlip(Transform):
    name: str = 'randomverticalflip'
    p: float = 0.5


@dataclass
class Resize(Transform):
    name: str = 'resize'
    size: int = DEFAULT_IMG_SIZE
    interpolation: Optional[str] = 'bilinear'


@dataclass
class AugmentationConfig:
    img_size: int = DEFAULT_IMG_SIZE
    recipe: List[Transform] = field(default_factory=list)
