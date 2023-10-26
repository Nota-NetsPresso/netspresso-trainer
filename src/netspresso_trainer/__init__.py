from pathlib import Path

from .train import parse_args_netspresso, set_arguments, train_with_yaml, train_with_config
from .cfg import *

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version