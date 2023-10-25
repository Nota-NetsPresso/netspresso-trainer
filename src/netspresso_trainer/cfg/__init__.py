from dataclasses import dataclass, field

from .augmentation import (
    AugmentationConfig,
    ClassificationAugmentationConfig,
    SegmentationAugmentationConfig,
    DetectionAugmentationConfig,
    ColorJitter
)
from .environment import EnvironmentConfig
from .logging import LoggingConfig

@dataclass
class TrainerConfig:
    augmentation: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    environment: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
