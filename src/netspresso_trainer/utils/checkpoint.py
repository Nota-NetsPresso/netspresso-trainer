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
