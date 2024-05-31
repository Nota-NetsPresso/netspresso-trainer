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
from abc import ABC, abstractmethod, abstractproperty
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from loguru import logger


class BaseCustomDataset(data.Dataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class, split, samples, transform, **kwargs):
        super(BaseCustomDataset, self).__init__()
        self.conf_data = conf_data
        self.conf_augmentation = conf_augmentation
        self.model_name = model_name

        self.transform = transform(conf_augmentation)
        self.samples = samples

        self._root = conf_data.path.root
        self._idx_to_class = idx_to_class
        self._num_classes = len(self._idx_to_class)
        self._split = split

        self.cache = False

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def cache_dataset(self, sampler, distributed):
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


class BaseHFDataset(data.Dataset):

    def __init__(self, conf_data, conf_augmentation, model_name, root, split, transform):
        super(BaseHFDataset, self).__init__()
        self.conf_data = conf_data
        self.conf_augmentation = conf_augmentation
        self.model_name = model_name
        self.transform = transform(conf_augmentation)
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


class BaseSampleLoader(ABC):
    def __init__(self, conf_data, train_valid_split_ratio):
        self.conf_data = conf_data
        self.train_valid_split_ratio = train_valid_split_ratio

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def load_id_mapping(self):
        raise NotImplementedError

    @abstractmethod
    def load_class_map(self, id_mapping):
        raise NotImplementedError

    def load_samples(self):
        assert self.conf_data.id_mapping is not None
        id_mapping = self.load_id_mapping()
        misc = self.load_class_map(id_mapping)

        train_samples, valid_samples, test_samples = self.load_split_samples()
        return train_samples, valid_samples, test_samples, misc

    def load_split_samples(self):
        exists_train = self.conf_data.path.train.image is not None
        exists_valid = self.conf_data.path.valid.image is not None
        exists_test = self.conf_data.path.test.image is not None

        train_samples = None
        valid_samples = None
        test_samples = None

        if exists_train:
            train_samples = self.load_data(split='train')
        if exists_valid:
            valid_samples = self.load_data(split='valid')
        if exists_test:
            test_samples = self.load_data(split='test')

        if not exists_valid and exists_train:
            logger.info(f"Validation set is not provided in config. Split automatically training set by {self.train_valid_split_ratio:.1f}:{1-self.train_valid_split_ratio:.1f}.")
            num_train_splitted = int(len(train_samples) * self.train_valid_split_ratio)
            train_samples, valid_samples = \
                data.random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                                  generator=torch.Generator().manual_seed(42))

        return train_samples, valid_samples, test_samples

    @abstractmethod
    def load_huggingface_samples(self):
        raise NotImplementedError
