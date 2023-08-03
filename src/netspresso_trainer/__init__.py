from pathlib import Path

from .trainer_common import set_arguments, trainer

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version