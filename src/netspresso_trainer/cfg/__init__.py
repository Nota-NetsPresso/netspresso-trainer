from dataclasses import dataclass, field
from typing import Optional

from .augmentation import (
    AugmentationConfig,
    ClassificationAugmentationConfig,
    SegmentationAugmentationConfig,
    DetectionAugmentationConfig,
    ColorJitter
)
from .data import *
from .model import *
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
    data: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    environment: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
    
    @property
    def epochs(self) -> int:
        return self.training.epochs
    
    @property
    def batch_size(self) -> int:
        return self.training.batch_size
    
    @property
    def num_workers(self) -> int:
        return self.environment.num_workers
    
    @epochs.setter
    def epochs(self, v: int) -> None:
        self.training.epochs = v

    @batch_size.setter
    def batch_size(self, v: int) -> None:
        self.training.batch_size = v
    
    @num_workers.setter
    def num_workers(self, v: int) -> None:
        self.environment.num_workers = v
