import os
import argparse
from typing import Literal

from omegaconf import OmegaConf, DictConfig

from .cfg import TrainerConfig
from .train_common import train_common

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def parse_args_netspresso():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")

    # -------- User arguments ----------------------------------------

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
        help="Config for training environment (# workers, etc.)")

    args_parsed, _ = parser.parse_known_args()

    return args_parsed


def set_arguments(args_parsed):
    conf_data = OmegaConf.load(args_parsed.data)
    conf_augmentation = OmegaConf.load(args_parsed.augmentation)
    conf_model = OmegaConf.load(args_parsed.model)
    conf_training = OmegaConf.load(args_parsed.training)
    conf_logging = OmegaConf.load(args_parsed.logging)
    conf_environment = OmegaConf.load(args_parsed.environment)

    conf = OmegaConf.create()
    conf.merge_with(conf_data)
    conf.merge_with(conf_augmentation)
    conf.merge_with(conf_model)
    conf.merge_with(conf_training)
    conf.merge_with(conf_logging)
    conf.merge_with(conf_environment)
    return conf

def train_with_yaml():
    args_parsed = parse_args_netspresso()
    conf = set_arguments(args_parsed)
    train_common(conf, log_level=args_parsed.log_level)

def train_with_config(config: TrainerConfig, log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'):
    conf: DictConfig = OmegaConf.create(config)
    OmegaConf.set_struct(conf, False)
    train_common(conf, log_level=log_level)


if __name__ == "__main__":
    train_with_yaml()