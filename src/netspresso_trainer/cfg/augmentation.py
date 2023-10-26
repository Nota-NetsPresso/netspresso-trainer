from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from omegaconf import MISSING, MissingMandatoryValue


@dataclass
class ColorJitter:
    brightness: Optional[float] = 0.25
    contrast: Optional[float] = 0.25
    saturation: Optional[float] = 0.25
    hue: Optional[float] = 0.1
    colorjitter_p: Optional[float] = 0.5


@dataclass
class AugmentationConfig:
    img_size: int = 256
    max_scale: Optional[int] = 1024
    min_scale: Optional[int] = None
    crop_size_h: Optional[int] = None
    crop_size_w: Optional[int] = None
    resize_ratio0: Optional[float] = None
    resize_ratiof: Optional[float] = None
    resize_add: Optional[float] = 1
    fliplr: Optional[float] = 0.5
    color_jitter: Optional[ColorJitter] = field(default_factory=lambda: ColorJitter())
    
    

@dataclass
class ClassificationAugmentationConfig(AugmentationConfig):
    resize_ratio0 = None
    resize_ratiof = None
    resize_add = None
    color_jitter = None


@dataclass
class SegmentationAugmentationConfig(AugmentationConfig):
    img_size = 512
    resize_ratio0 = 1.0
    resize_ratiof = 1.5
    
    def __post_init__(self):
        # variable interpolation
        if self.min_scale is None:
            self.min_scale = self.img_size
        if self.crop_size_h is None:
            self.crop_size_h = self.img_size
        if self.crop_size_w is None:
            self.crop_size_w = self.img_size
    

@dataclass
class DetectionAugmentationConfig(AugmentationConfig):
    img_size = 512
    max_scale = 2048
    min_scale = 768
    resize_ratio0: 0.5
    resize_ratiof: 2.0
    resize_add: 1
    
    def __post_init__(self):
        # variable interpolation
        if self.crop_size_h is None:
            self.crop_size_h = self.img_size
        if self.crop_size_w is None:
            self.crop_size_w = self.img_size