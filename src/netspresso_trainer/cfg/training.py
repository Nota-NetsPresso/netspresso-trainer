from dataclasses import dataclass, field
from typing import Dict

from omegaconf import MISSING, MissingMandatoryValue


@dataclass
class ScheduleConfig:
    seed: int = 1
    sched: str = "cosine"
    min_lr: float = 1e-6
    warmup_bias_lr: float = 1e-5
    warmup_epochs: int = 5
    iters_per_phase: int = 30
    sched_power: float = 1.0
    epochs: int = 3
    batch_size: int = 8
    optimizer: Dict = field(default_factory=lambda: {
        "name": "adamw",
        "lr": 6e-5,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0005,
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
