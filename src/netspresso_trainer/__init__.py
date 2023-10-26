from pathlib import Path

from .trainer import parse_args_netspresso, set_arguments, train_with_yaml, train_with_config
from .cfg import *

train = train_with_config

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version