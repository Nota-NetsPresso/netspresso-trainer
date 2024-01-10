from dataclasses import dataclass, field
from typing import Dict

from omegaconf import MISSING, MissingMandatoryValue


@dataclass
class ScheduleConfig:
    epochs: int = 3
    batch_size: int = 8
    optimizer: Dict = field(default_factory=lambda: {
        "name": "adamw",
        "lr": 6e-5,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0005,
    })
    scheduler: Dict = field(default_factory=lambda: {
        "name": "cosine_no_sgdr",
        "warmup_epochs": 5,
        "warmup_bias_lr": 1e-5,
        "min_lr": 0.,
    })


@dataclass
class ClassificationScheduleConfig(ScheduleConfig):
    batch_size: int = 32


@dataclass
class SegmentationScheduleConfig(ScheduleConfig):
    pass


@dataclass
class DetectionScheduleConfig(ScheduleConfig):
    pass
