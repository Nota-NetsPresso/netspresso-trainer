import os
from pathlib import Path
from typing import List, Literal, Optional, Union

from netspresso_trainer.trainer_common import train_common
from netspresso_trainer.trainer_util import (
    set_arguments,
    train_with_yaml_impl,
)
from .utils.engine_utils import parse_args_netspresso, parse_gpu_ids

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


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


def train_cli() -> None:
    args_parsed = parse_args_netspresso(with_gpus=True, isTrain=True)

    logging_dir: Path = train_with_yaml_impl(
        gpus=args_parsed.gpus,
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        training=args_parsed.training,
        logging=args_parsed.logging,
        environment=args_parsed.environment,
        log_level=args_parsed.log_level
    )

    return logging_dir


def train_cli_without_additional_gpu_check() -> None:
    args_parsed = parse_args_netspresso(with_gpus=False, isTrain=True)

    conf = set_arguments(
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        training=args_parsed.training,
        logging=args_parsed.logging,
        environment=args_parsed.environment
    )

    train_common(
        conf,
        task=args_parsed.task,
        model_name=args_parsed.model_name,
        is_graphmodule_training=args_parsed.is_graphmodule_training,
        logging_dir=args_parsed.logging_dir,
        log_level=args_parsed.log_level
    )

if __name__ == "__main__":

    # Execute by `run_distributed_training_script`
    train_cli_without_additional_gpu_check()
