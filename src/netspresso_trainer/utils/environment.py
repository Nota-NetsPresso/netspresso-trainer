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

import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

__all__ = ['set_device', 'get_device']


def set_device(seed):
    # Torch settings
    cudnn.enabled = False

    # Reproducibility (except cudnn deterministicity)
    if seed is not None:
        torch.manual_seed(seed)  # for cpu
        torch.cuda.manual_seed(seed)  # for gpu
        random.seed(seed)
        np.random.seed(seed)

    # prepare device
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # distributed
    distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    world_size = 1
    rank = 0  # global rank
    if distributed:
        assert seed is not None and cudnn.benchmark is False, "distributed training requires reproducibility"
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        devices = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
    assert rank >= 0

    return distributed, world_size, rank, devices


def get_device(x: Union[torch.Tensor, nn.Module]):
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, nn.Module):
        return next(x.parameters()).device
    raise RuntimeError(f'{type(x)} do not have `device`')
