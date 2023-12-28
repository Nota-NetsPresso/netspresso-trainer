# import logging
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ['set_logger', 'yaml_for_logging']

OUTPUT_ROOT_DIR = "./outputs"
LEVELNO_TO_LEVEL_NAME = {
    0: "NOTSET",
    10: "DEBUG",
    20: "INFO",
    30: "WARNING",
    40: "ERROR",
    50: "CRITICAL",
}

def rank_filter(record):
    try:
        return dist.get_rank() == 0
    except RuntimeError:  # Default process group has not been initialized, please make sure to call init_process_group.
        return True

def get_format(level: str, distributed: bool = False):
    debug_and_multi_gpu = (level == 'DEBUG' and distributed)
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    if debug_and_multi_gpu:
        fmt = f"[GPU:{dist.get_rank()}] " + fmt

    only_rank_zero = not debug_and_multi_gpu
    return fmt, only_rank_zero

def add_stream_handler(level: str, distributed: bool):
    fmt, only_rank_zero = get_format(level, distributed=distributed)
    logger.add(sys.stderr, level=level, format=fmt, filter=rank_filter if only_rank_zero else "")



def add_file_handler(log_filepath: str, distributed: bool):
    level = LEVELNO_TO_LEVEL_NAME[logger._core.min_level]
    fmt, only_rank_zero = get_format(level, distributed=distributed)
    logger.add(log_filepath, level=level, format=fmt, filter=rank_filter if only_rank_zero else "", enqueue=True)



def _custom_logger(level: str, distributed: bool):
    logger.remove()
    add_stream_handler(level, distributed)

    return logger



def set_logger(level: str = 'INFO', distributed=False):
    try:
        time.tzset()
    except AttributeError as e:
        print(e)
        print("Skipping timezone setting.")
    _level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = level.upper()
    _custom_logger(_level, distributed)

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
    set_logger(level='DEBUG')
