from pathlib import Path

from .cfg import TrainerConfig
from .trainer import export_config_as_yaml, parse_args_netspresso, set_arguments, train_with_config, train_with_yaml

train = train_with_config

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version