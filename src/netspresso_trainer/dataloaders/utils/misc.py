# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

import csv
import json
import re
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def get_detection_label(label_file: Path):
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


def load_classification_class_map(labels_path: Optional[Union[str, Path]]):
    # Assume the `map_or_filename` is path for csv label file
    assert labels_path.exists(), f"Cannot locate specified class map file {labels_path}!"
    class_map_ext = labels_path.suffix.lower()
    assert class_map_ext == '.csv', f"Unsupported class map file extension ({class_map_ext})!"

    with open(labels_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        file_to_idx = {row['image_id']: int(row['class']) for row in reader}

    return file_to_idx


def is_file_dict(image_dir: Union[Path, str], file_or_dir_to_idx):
    image_dir = Path(image_dir)
    candidate_name = list(file_or_dir_to_idx.keys())[0]
    file_or_dir: Path = image_dir / candidate_name
    if file_or_dir.exists():
        return file_or_dir.is_file()

    file_candidates = list(image_dir.glob(f"{candidate_name}.*"))
    assert len(file_candidates) != 0, f"Unknown label format! Is there any something file like {file_or_dir} ?"

    return True


def as_tuple(tuple_string: str) -> Tuple:
    tuple_string = tuple_string.replace("(", "")
    tuple_string = tuple_string.replace(")", "")
    tuple_value = tuple(map(int, tuple_string.split(",")))
    return tuple_value
