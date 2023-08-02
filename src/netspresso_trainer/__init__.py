from pathlib import Path

from .train_common import train, set_arguments

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version