from pathlib import Path

from .cfg import TrainerConfig
from .trainer_cli import parse_args_netspresso, set_arguments, train_cli
from .trainer_inline import export_config_as_yaml, train_with_config

train = train_with_config

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version