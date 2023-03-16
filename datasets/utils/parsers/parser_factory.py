from datasets.utils.parsers.parser_image_folder import ParserImageFolder
import os
from pathlib import Path

_RECOMMEND_DATASET_DIR = "./data"


def create_parser(name, root, split='train', class_map=_RECOMMEND_DATASET_DIR, **kwargs):
    # name = name.lower()

    _root = Path(root) / split

    assert _root.exists() and _root.is_dir(), f"No such directory {_root}!"

    parser = ParserImageFolder(_root, class_map=class_map, **kwargs)
    return parser
