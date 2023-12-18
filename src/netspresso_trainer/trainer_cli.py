import os
from pathlib import Path

from netspresso_trainer.trainer_util import train_with_yaml_impl, parse_args_netspresso

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def train_cli() -> None:
    args_parsed = parse_args_netspresso(with_gpus=True)

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