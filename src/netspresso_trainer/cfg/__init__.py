from dataclasses import dataclass, field
from typing import Optional

from .augmentation import (
    AugmentationConfig,
    ClassificationAugmentationConfig,
    SegmentationAugmentationConfig,
    DetectionAugmentationConfig,
    ColorJitter
)
from .training import (
    TrainingConfig,
    ClassificationTrainingConfig,
    SegmentationTrainingConfig,
    DetectionTrainingConfig
)
from .environment import EnvironmentConfig
from .logging import LoggingConfig

@dataclass
class TrainerConfig:
    augmentation: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    environment: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
