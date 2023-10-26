from dataclasses import dataclass, field
from typing import Optional, Dict, Any

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
    ScheduleConfig,
    ClassificationScheduleConfig,
    SegmentationScheduleConfig,
    DetectionScheduleConfig
)
from .environment import EnvironmentConfig
from .logging import LoggingConfig

from omegaconf import MISSING, MissingMandatoryValue

_AUGMENTATION_CONFIG_TYPE_DICT = {
    'classification': ClassificationAugmentationConfig,
    'segmentation': SegmentationAugmentationConfig,
    'detection': DetectionAugmentationConfig
}

_TRAINING_CONFIG_TYPE_DICT = {
    'classification': ClassificationScheduleConfig,
    'segmentation': SegmentationScheduleConfig,
    'detection': DetectionScheduleConfig
}

@dataclass
class TrainerConfig:
    task: str = MISSING
    auto: bool = False
    augmentation: Optional[AugmentationConfig] = None
    training: Optional[ScheduleConfig] = None
    data: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
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
        
    def __post_init__(self):
        assert self.task in ['classification', 'segmentation', 'detection']
        self.data.task = self.task
        self.model.task = self.task
        
        if self.auto:
            if self.augmentation is None:
                self.augmentation = _AUGMENTATION_CONFIG_TYPE_DICT[self.task]()
            if self.training is None:
                self.training = _TRAINING_CONFIG_TYPE_DICT[self.task]()