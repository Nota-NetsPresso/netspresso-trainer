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
            args_train,
            root,
            split
    ):
        super(BaseCustomDataset, self).__init__()
        self.args = args_train
        self._root = root
        self._split = split

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    def root(self):
        return self._root

    @property
    def mode(self):
        return self._split
