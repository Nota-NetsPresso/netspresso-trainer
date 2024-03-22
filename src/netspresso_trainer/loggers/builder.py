from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from .base import TrainingLogger


def build_logger(conf, task: str, model_name: str, step_per_epoch: int, class_map: Dict[int, str], num_sample_images: int, result_dir: Union[Path, str], epoch: Optional[int] = None):
    training_logger = TrainingLogger(conf,
                                     task=task, model=model_name,
                                     step_per_epoch=step_per_epoch,
                                     class_map=class_map, num_sample_images=num_sample_images,
                                     result_dir=result_dir,
                                     epoch=epoch)

    return training_logger
