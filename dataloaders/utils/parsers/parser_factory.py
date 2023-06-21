from dataloaders.utils.parsers.parser_image_folder import ParserImageFolder
import os
from pathlib import Path


def create_parser(name, root, split='train', class_map=None, **kwargs):
    # name = name.lower()

    _root = Path(root) / split

    assert _root.exists() and _root.is_dir(), f"No such directory {_root}!"

    parser = ParserImageFolder(_root, class_map=class_map, **kwargs)
    return parser
