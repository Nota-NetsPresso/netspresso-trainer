from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING, MissingMandatoryValue

from .augmentation import (
    AugmentationConfig,
    ClassificationAugmentationConfig,
    ColorJitter,
    DetectionAugmentationConfig,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    SegmentationAugmentationConfig,
)
from .data import (
    DatasetConfig,
    ExampleBeansDataset,
    ExampleChessDataset,
    ExampleCocoyoloDataset,
    ExampleSidewalkDataset,
    ExampleSkincancerDataset,
    ExampleTrafficsignDataset,
    ExampleVoc12CustomDataset,
    ExampleVoc12Dataset,
    ExampleWikiartDataset,
    ExampleXrayDataset,
    HuggingFaceClassificationDatasetConfig,
    HuggingFaceSegmentationDatasetConfig,
    LocalClassificationDatasetConfig,
    LocalDetectionDatasetConfig,
    LocalSegmentationDatasetConfig,
)
from .environment import EnvironmentConfig
from .logging import LoggingConfig
from .model import (
    ClassificationEfficientFormerModelConfig,
    ClassificationMixNetLargeModelConfig,
    ClassificationMixNetMediumModelConfig,
    ClassificationMixNetSmallModelConfig,
    ClassificationMobileNetV3ModelConfig,
    ClassificationMobileViTModelConfig,
    ClassificationResNetModelConfig,
    ClassificationViTModelConfig,
    DetectionEfficientFormerModelConfig,
    DetectionYoloXModelConfig,
    ModelConfig,
    PIDNetModelConfig,
    SegmentationEfficientFormerModelConfig,
    SegmentationMixNetLargeModelConfig,
    SegmentationMixNetMediumModelConfig,
    SegmentationMixNetSmallModelConfig,
    SegmentationMobileNetV3ModelConfig,
    SegmentationResNetModelConfig,
    SegmentationSegFormerModelConfig,
)
from .training import ClassificationScheduleConfig, DetectionScheduleConfig, ScheduleConfig, SegmentationScheduleConfig

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
    task: str = field(default=MISSING, metadata={"omegaconf_ignore": True})
    auto: bool = field(default=False, metadata={"omegaconf_ignore": True})
    data: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    augmentation: Optional[AugmentationConfig] = None
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    training: Optional[ScheduleConfig] = None
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
    environment: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())

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
