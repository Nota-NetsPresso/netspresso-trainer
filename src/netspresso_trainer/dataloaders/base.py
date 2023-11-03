import os
from abc import ABC, abstractmethod, abstractproperty
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data


class BaseCustomDataset(data.Dataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class, split, samples, transform, with_label, **kwargs):
        super(BaseCustomDataset, self).__init__()
        self.conf_data = conf_data
        self.conf_augmentation = conf_augmentation
        self.model_name = model_name

        self.transform = transform
        self.samples = samples

        self._root = conf_data.path.root
        self._idx_to_class = idx_to_class
        self._num_classes = len(self._idx_to_class)
        self._split = split
        self._with_label = with_label

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self._idx_to_class

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

    def __init__(self, conf_data, conf_augmentation, model_name, root, split, with_label):
        super(BaseHFDataset, self).__init__()
        self.conf_data = conf_data
        self.conf_augmentation = conf_augmentation
        self.model_name = model_name
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
    def __init__(self, conf_data, train_valid_split_ratio):
        self.conf_data = conf_data
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
