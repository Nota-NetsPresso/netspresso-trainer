import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from netspresso_trainer.cfg import TrainerConfig
from netspresso_trainer.trainer_util import (
    get_gpus_from_parser_and_config,
    parse_gpu_ids,
    train_with_config_impl,
    train_with_yaml_impl,
)

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def export_config_as_yaml(config: TrainerConfig) -> str:
    conf: DictConfig = OmegaConf.create(config)
    return OmegaConf.to_yaml(conf)


def train_with_config(
    config: TrainerConfig,
    gpus: Optional[str] = None,
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
) -> None:

    gpus: Union[List, int] = parse_gpu_ids(gpus)

    logging_dir: Path = train_with_config_impl(
        gpus=gpus,
        config=config,
        log_level=log_level
    )

    return logging_dir


def train_with_yaml(
    data: Union[Path, str],
    augmentation: Union[Path, str],
    model: Union[Path, str], training: Union[Path, str],
    logging: Union[Path, str], environment: Union[Path, str],
    gpus: Optional[str] = None, log_level: str = LOG_LEVEL
):

    gpus: Union[List, int] = parse_gpu_ids(gpus)

    logging_dir: Path = train_with_yaml_impl(
        gpus=gpus,
        data=data,
        augmentation=augmentation,
        model=model,
        training=training,
        logging=logging,
        environment=environment,
        log_level=log_level
    )

    return logging_dir
