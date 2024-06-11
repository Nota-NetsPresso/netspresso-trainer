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

import warnings
from pathlib import Path
from typing import Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _validate_file(file_path: Path):
    file_path = Path(file_path)

    extension = file_path.suffix
    supporting_extensions = ['.pth', '.pt', '.ckpt', '.safetensors']
    assert extension in supporting_extensions, f"The checkpoint format should be one of {supporting_extensions}! The given file is {file_path}."
    return file_path, extension


def load_checkpoint(f: Union[str, Path]):
    file_path, extension = _validate_file(f)

    if extension == '.safetensors':
        state_dict = {}
        with safe_open(str(file_path), framework="pt", device='cpu') as f:
            state_dict_keys = f.keys()
            for k in state_dict_keys:
                state_dict[k] = f.get_tensor(k)
        return state_dict

    state_dict = torch.load(file_path, map_location='cpu')
    return state_dict


def save_checkpoint(obj_dict, f: Union[str, Path]) -> None:
    file_path, extension = _validate_file(f)

    if extension == '.safetensors':
        save_file(obj_dict, str(file_path))
        return

    torch.save(obj_dict, file_path)
