import os
import logging
from abc import abstractmethod
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from datasets.utils.parsers import create_parser

_logger = logging.getLogger(__name__)

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
        
    @abstractmethod
    def __getitem__(self, index):
        pass
        
    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)

