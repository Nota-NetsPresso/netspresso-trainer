import logging
import time
from typing import Literal

import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ['set_logger', 'yaml_for_logging']


class RankFilter(logging.Filter):
    def filter(self, record):
        try:
            return dist.get_rank() == 0
        except RuntimeError:  # Default process group has not been initialized, please make sure to call init_process_group.
            return True


def _custom_logger(name: str, level: str, distributed: bool):
    fmt_date = '%Y-%m-%d_%T %Z'
    debug_and_multi_gpu = (level == 'DEBUG' and distributed)
    if debug_and_multi_gpu:
        fmt = f'[GPU:{dist.get_rank()}] %(asctime)s | %(levelname)s\t\t| %(funcName)s:<%(filename)s>:%(lineno)s >>> %(message)s'
    else:
        fmt = '%(asctime)s | %(levelname)s\t\t| %(funcName)s:<%(filename)s>:%(lineno)s >>> %(message)s'
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        handler = logging.StreamHandler()

        formatter = logging.Formatter(fmt, fmt_date)
        handler.setFormatter(formatter)
        if distributed:
            handler.addFilter(RankFilter())

        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


def set_logger(logger_name="netspresso_trainer", level: str = 'INFO', distributed=False):
    try:
        time.tzset()
    except AttributeError as e:
        print(e)
        print("Skipping timezone setting.")
    _level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = level.upper()
    _custom_logger(logger_name, _level, distributed)
    
    logger = logging.getLogger(logger_name)
    if _level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif _level == 'INFO':
        logger.setLevel(logging.INFO)
    elif _level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif _level == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif _level == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)
    return logger


def _yaml_for_logging(config: DictConfig) -> DictConfig:
    # TODO: better configuration logging
    list_maximum_index = 2
    new_config = OmegaConf.create()
    for k, v in config.items():
        if isinstance(v, DictConfig):
            new_config.update({k: _yaml_for_logging(v)})
        elif isinstance(v, ListConfig):
            new_config.update({k: list(map(str, v[:list_maximum_index])) + ['...']})
        else:
            new_config.update({k: v})
    return new_config


def yaml_for_logging(config: DictConfig):
    config_summarized = OmegaConf.create(_yaml_for_logging(config))
    return OmegaConf.to_yaml(config_summarized)


if __name__ == '__main__':
    set_logger(__name__, level='DEBUG')
