import os
import logging
from abc import abstractmethod
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from datasets.parsers import create_parser
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50

class BaseCustomDataset(data.Dataset):

    def __init__(
            self,
            args,
            root,
            parser=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        super(BaseCustomDataset, self).__init__()
        self.args = args
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        
        self.class_map = args.class_map if Path(args.class_map).exists() else None

    @abstractmethod
    def __getitem__(self, index):
        pass
        
    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)
    
def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x
    
def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False
        

class PrefetchLoader:

    def __init__(
            self,
            loader,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            channels=3,
            fp16=False):

        mean = expand_to_chs(mean, channels)
        std = expand_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.type(torch.int64).cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def transforms_config(dataset: str, train: bool, cfg = None):
    transf_config = {}

    if train:
        transf_config = {
            'hflip': 0.5,
            'vflip': 0,
            'mean': IMAGENET_DEFAULT_MEAN, 
            'std': IMAGENET_DEFAULT_STD
        }
    else:
        transf_config = {
            'mean': IMAGENET_DEFAULT_MEAN, 
            'std': IMAGENET_DEFAULT_STD
        }

    if cfg is not None:
        for key in cfg.keys():
            transf_config[key] = cfg[key]

    return transf_config
