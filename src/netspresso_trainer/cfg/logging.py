from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from omegaconf import MISSING, MissingMandatoryValue


@dataclass
class LoggingConfig:
    project_id: Optional[str] = None
    output_dir: Union[Path, str] = "./outputs"
    tensorboard: bool = True
    csv: bool = False
    image: bool = True
    stdout: bool = True
    save_optimizer_state: bool = True
    validation_epoch: int = 10
    save_checkpoint_epoch: Optional[int] = None

    def __post_init__(self):
        if self.save_checkpoint_epoch is None:
            self.save_checkpoint_epoch = self.validation_epoch
