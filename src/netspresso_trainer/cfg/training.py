from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

from omegaconf import MISSING, MissingMandatoryValue


@dataclass
class TrainingConfig:
    seed: int = 1
    opt: str = "adamw"
    lr: float = 6e-5
    momentum: float =  0.937
    weight_decay: float = 0.0005
    sched: str = "cosine"
    min_lr: float = 1e-6
    warmup_bias_lr: float = 1e-5
    warmup_epochs: int = 5
    iters_per_phase: int = 30
    sched_power: float = 1.0
    epochs: int = 3
    batch_size: int = 8


@dataclass
class ClassificationTrainingConfig(TrainingConfig):
    batch_size: int = 32


@dataclass
class SegmentationTrainingConfig(TrainingConfig):
    pass


@dataclass
class DetectionTrainingConfig(TrainingConfig):
    pass