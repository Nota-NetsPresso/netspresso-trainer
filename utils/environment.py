import os
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np


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
        assert seed is not None and cudnn.benchmark == False, "distributed training requires reproducibility"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        rank = torch.distributed.get_rank()
        devices = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        world_size = torch.distributed.get_world_size()
    assert rank >= 0

    return distributed, world_size, rank, devices
