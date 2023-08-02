from pathlib import Path

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version