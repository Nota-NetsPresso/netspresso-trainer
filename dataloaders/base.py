import os
import logging
from abc import abstractmethod, abstractproperty
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

_logger = logging.getLogger(__name__)


class BaseCustomDataset(data.Dataset):

    def __init__(
            self,
            args,
            root,
            split
    ):
        super(BaseCustomDataset, self).__init__()
        self.args = args
        self._root = root
        self._split = split

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractproperty
    def num_classes(self):
        pass

    @abstractproperty
    def class_map(self):
        pass

    @property
    def root(self):
        return self._root

    @property
    def mode(self):
        return self._split
    
    
class BaseHFDataset(data.Dataset):

    def __init__(
            self,
            args,
            root,
            split
    ):
        super(BaseHFDataset, self).__init__()
        self.args = args
        self._root = root
        self._split = split

    def _load_dataset(self, root, subset_name=None, cache_dir=None):         
        from datasets import load_dataset
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
        total_dataset = load_dataset(root, name=subset_name, cache_dir=cache_dir)
        return total_dataset

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractproperty
    def num_classes(self):
        pass

    @abstractproperty
    def class_map(self):
        pass

    @property
    def root(self):
        return self._name

    @property
    def mode(self):
        return self._split