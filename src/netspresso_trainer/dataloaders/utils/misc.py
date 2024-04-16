import json
import re
from itertools import repeat
from pathlib import Path

import numpy as np


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_label(label_file: Path):
    target = Path(label_file).read_text()

    if target == '': # target label can be empty string
        target_array = np.zeros((0, 5))
    else:
        try:
            target_array = np.array([list(map(float, box.split(' '))) for box in target.split('\n') if box.strip()])
        except ValueError as e:
            print(target)
            raise e

    label, boxes = target_array[:, 0], target_array[:, 1:]
    label = label[..., np.newaxis]
    return label, boxes
