from dataclasses import dataclass, field
from typing import Dict, Optional

from omegaconf import MISSING, MissingMandatoryValue


@dataclass
class EMA:
    name: str = MISSING
    decay: float = MISSING


@dataclass
class ConstantDecayEMA(EMA):
    name: str = 'constant_decay'
    decay: float = 0.9999


@dataclass
class ExpDecayEMA(EMA):
    name: str = 'exp_decay'
    decay: float = 0.9999
    beta: float = 15


@dataclass
class ScheduleConfig:
    epochs: int = 3
    batch_size: int = 8
    ema: Optional[EMA] = None
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
        "end_epoch": 3,
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


@dataclass
class PoseEstimationScheduleConfig(ScheduleConfig):
    pass