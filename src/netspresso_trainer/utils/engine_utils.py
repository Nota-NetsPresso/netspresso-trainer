import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from netspresso_trainer.trainer_common import train_common

OUTPUT_ROOT_DIR = "./outputs"
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args_netspresso(with_gpus=False, isTrain=True):

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")

    # -------- User arguments ----------------------------------------

    if with_gpus:
        parser.add_argument(
            '--gpus', type=parse_gpu_ids, default="",
            dest='gpus',
            help='GPU device indices (comma-separated)')

    parser.add_argument(
        '--data', type=str, required=True,
        dest='data',
        help="Config for dataset information")

    parser.add_argument(
        '--augmentation', type=str, required=True,
        dest='augmentation',
        help="Config for data augmentation")

    parser.add_argument(
        '--model', type=str, required=True,
        dest='model',
        help="Config for the model architecture")

    if isTrain:
        parser.add_argument(
            '--training', type=str, required=True,
            dest='training',
            help="Config for training options")

    parser.add_argument(
        '--logging', type=str, default='config/logging.yaml',
        dest='logging',
        help="Config for logging options")

    parser.add_argument(
        '--environment', type=str, default='config/environment.yaml',
        dest='environment',
        help="Config for training environment (# workers, etc.)")

    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default=LOG_LEVEL,
        dest='log_level',
        help="Logging level in training process")

    args_parsed, _ = parser.parse_known_args()

    return args_parsed


def parse_gpu_ids(gpu_arg: str) -> Optional[Union[List, int]]:
    """Parse comma-separated GPU IDs and return as a list of integers."""
    if gpu_arg is None or str(gpu_arg) in ["", "None"]:
        return None

    try:
        gpu_ids = [int(id) for id in gpu_arg.split(',')]

        if len(gpu_ids) == 1:  # Single GPU
            return gpu_ids[0]

        gpu_ids = sorted(gpu_ids)
        return gpu_ids
    except ValueError as e:
        raise argparse.ArgumentTypeError('Invalid GPU IDs. Please provide comma-separated integers.') from e


def set_arguments(
    data: Union[Path, str],
    augmentation: Union[Path, str],
    model: Union[Path, str],
    logging: Union[Path, str],
    environment: Union[Path, str],
    training: Optional[Union[Path, str]] = None,
) -> DictConfig:

    conf_data = OmegaConf.load(data)
    conf_augmentation = OmegaConf.load(augmentation)
    conf_model = OmegaConf.load(model)
    if training:
        conf_training = OmegaConf.load(training)
    conf_logging = OmegaConf.load(logging)
    conf_environment = OmegaConf.load(environment)

    conf = OmegaConf.create()
    conf.merge_with(conf_data)
    conf.merge_with(conf_augmentation)
    conf.merge_with(conf_model)
    if training:
        conf.merge_with(conf_training)
    conf.merge_with(conf_logging)
    conf.merge_with(conf_environment)

    return conf


def get_gpu_from_config(conf_environment: DictConfig) -> Optional[Union[List, int]]:
    conf_environment_gpus = str(conf_environment.gpus) if hasattr(conf_environment, 'gpus') else None
    return parse_gpu_ids(conf_environment_gpus)


def get_gpus_from_parser_and_config(
    gpus: Optional[Union[List, int]],
    conf_environment: DictConfig
) -> Union[List, int]:
    conf_environment_gpus = get_gpu_from_config(conf_environment)
    if gpus is None:
        if conf_environment_gpus is None:
            return 0  # Try use the 'cuda:0'
        return conf_environment_gpus
    return gpus
