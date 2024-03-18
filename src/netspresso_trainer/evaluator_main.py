import os
from pathlib import Path
from typing import List, Literal, Optional, Union

from netspresso_trainer.evaluator_common import evaluation_common
from netspresso_trainer.evaluator_util import (
    evaluation_with_yaml_impl,
)
from .utils.engine_utils import parse_args_netspresso

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def evaluation_cli() -> None:
    args_parsed = parse_args_netspresso(with_gpus=True, isTrain=False)

    logging_dir: Path = evaluation_with_yaml_impl(
        gpus=args_parsed.gpus,
        data=args_parsed.data,
        augmentation=args_parsed.augmentation,
        model=args_parsed.model,
        logging=args_parsed.logging,
        environment=args_parsed.environment,
        log_level=args_parsed.log_level
    )

    return logging_dir
