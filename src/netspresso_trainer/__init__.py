from pathlib import Path

from .trainer_common import parse_args_netspresso, set_arguments, trainer

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version