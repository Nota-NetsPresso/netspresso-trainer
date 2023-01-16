import os
import logging
from abc import abstractmethod
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from datasets.utils.parsers import create_parser
from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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
