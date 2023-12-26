from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import MISSING, MissingMandatoryValue

DEFAULT_IMG_SIZE = 256


@dataclass
class Transform:
    name: str = MISSING


@dataclass
class AugmentationConfig:
    img_size: int = DEFAULT_IMG_SIZE
    transforms: List[Transform] = field(default_factory=lambda: [
        Transform()
    ])
    mix_transforms: Optional[List[Transform]] = None


@dataclass
class ColorJitter(Transform):
    name: str = 'colorjitter'
    brightness: Optional[float] = 0.25
    contrast: Optional[float] = 0.25
    saturation: Optional[float] = 0.25
    hue: Optional[float] = 0.1
    p: Optional[float] = 0.5


@dataclass
class Pad(Transform):
    name: str = 'pad'
    padding: Union[int, List] = 0
    fill: Union[int, List] = 0
    padding_mode: str = 'constant'


@dataclass
class RandomCrop(Transform):
    name: str = 'randomcrop'
    size: int = DEFAULT_IMG_SIZE


@dataclass
class RandomResizedCrop(Transform):
    name: str = 'randomresizedcrop'
    size: int = DEFAULT_IMG_SIZE
    scale: List = field(default_factory=lambda: [0.08, 1.0])
    ratio: List = field(default_factory=lambda: [0.75, 1.33])
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
    max_size: Optional[int] =  None


class TrivialAugmentWide(Transform):
    name: str = 'trivialaugmentwide'
    num_magnitude_bins: int = 31
    interpolation: str = 'bilinear'
    fill: Optional[Union[int, List]] = None


@dataclass
class RandomMixup(Transform):
    name: str = 'mixup'
    alpha: float = 0.2
    p: float = 1.0
    inplace: bool = False


@dataclass
class RandomCutmix(Transform):
    name: str = 'cutmix'
    alpha: float = 1.0
    p: float = 1.0
    inplace: bool = False


@dataclass
class ClassificationAugmentationConfig(AugmentationConfig):
    img_size: int = 256
    transforms: List[Transform] = field(default_factory=lambda: [
        RandomResizedCrop(size=256),
        RandomHorizontalFlip()
    ])
    mix_transforms: List[Transform] = field(default_factory=lambda: [
        RandomCutmix(),
    ])


@dataclass
class SegmentationAugmentationConfig(AugmentationConfig):
    img_size: int = 512
    transforms: List[Transform] = field(default_factory=lambda: [
        RandomResizedCrop(size=512),
        RandomHorizontalFlip(),
        ColorJitter()
    ])


@dataclass
class DetectionAugmentationConfig(AugmentationConfig):
    img_size: int = 512
    transforms: List[Transform] = field(default_factory=lambda: [
        Resize(size=512)
    ])
