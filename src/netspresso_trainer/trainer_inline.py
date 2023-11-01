import os
from pathlib import Path
from typing import List, Literal, Union

from omegaconf import DictConfig, OmegaConf

from netspresso_trainer.cfg import TrainerConfig
from netspresso_trainer.trainer_cli import parse_gpu_ids, train_with_yaml_impl
from netspresso_trainer.trainer_common import train_common

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def set_struct_recursive(conf: DictConfig, value: bool) -> None:
    OmegaConf.set_struct(conf, value)
    
    for _, conf_value in conf.items():
        if isinstance(conf_value, DictConfig):
            set_struct_recursive(conf_value, value)


def export_config_as_yaml(config: TrainerConfig) -> str:
    conf: DictConfig = OmegaConf.create(config)
    return OmegaConf.to_yaml(conf)


def train_with_config(config: TrainerConfig, log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO') -> None:
    conf: DictConfig = OmegaConf.create(config)
    set_struct_recursive(conf, False)
    train_common(conf, log_level=log_level)


def train_with_yaml(gpus: str, data: Union[Path, str], augmentation: Union[Path, str],
                    model: Union[Path, str], training: Union[Path, str],
                    logging: Union[Path, str], environment: Union[Path, str], log_level: str = LOG_LEVEL):
    
    gpus: Union[List, int] = parse_gpu_ids(gpus)
    
    train_with_yaml_impl(
        gpus=gpus,
        data=data,
        augmentation=augmentation,
        model=model,
        training=training,
        logging=logging,
        environment=environment,
        log_level=log_level
    )