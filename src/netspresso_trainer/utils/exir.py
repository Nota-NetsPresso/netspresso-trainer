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
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from .environment import get_device

__all__ = ['save_exir']


def save_exir(model: nn.Module, f: Union[str, Path], sample_input: Tensor):
    if not hasattr(torch, 'export'):
        logger.warning("Current torch version does not support torch.export. Please upgrade torch.")
        return
    sample_input = sample_input.to(get_device(model))
    exported_program = torch.export.export(model, (sample_input, ))
    torch.export.save(exported_program, f)
    return exported_program
