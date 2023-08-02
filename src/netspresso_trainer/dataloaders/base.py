import os
from abc import ABC, abstractmethod, abstractproperty
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from utils.logger import set_logger

logger = set_logger('dataloaders', level=os.getenv('LOG_LEVEL', 'INFO'))

class BaseCustomDataset(data.Dataset):

    def __init__(
            self,
            args,
            root,
            split,
            with_label
    ):
        super(BaseCustomDataset, self).__init__()
        self.args = args
        self._root = root
        self._split = split
        self._with_label = with_label

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
    
    @property
    def with_label(self):
        return self._with_label
    
    
class BaseHFDataset(data.Dataset):

    def __init__(
            self,
            args,
            root,
            split,
            with_label
    ):
        super(BaseHFDataset, self).__init__()
        self.args = args
        self._root = root
        self._split = split
        self._with_label = with_label

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
    
    @property
    def with_label(self):
        return self._with_label


class BaseDataSampler(ABC):
    def __init__(self, args_data, train_valid_split_ratio):
        self.args_data = args_data
        self.train_valid_split_ratio = train_valid_split_ratio
    
    @abstractmethod
    def load_data(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_samples(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_huggingface_samples(self):
        raise NotImplementedError