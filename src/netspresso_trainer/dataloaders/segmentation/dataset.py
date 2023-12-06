import json
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import random_split

from ..base import BaseDataSampler
from ..utils.constants import IMG_EXTENSIONS
from ..utils.misc import natural_key


def as_tuple(tuple_string: str) -> Tuple:
    tuple_string = tuple_string.replace("(", "")
    tuple_string = tuple_string.replace(")", "")
    tuple_value = tuple(map(int, tuple_string.split(",")))
    return tuple_value


def load_custom_class_map(id_mapping: Union[ListConfig, DictConfig]) -> Tuple[Dict[int, str], Dict[Union[int, Tuple], int]]:
    if isinstance(id_mapping, ListConfig):
        assert isinstance(id_mapping[0], str), f"Unknown type for class name! {type(id_mapping[0])}"
        idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
        label_value_to_idx = {k: k for k in idx_to_class}
        return idx_to_class, label_value_to_idx

    idx_to_class: Dict[int, str] = {}
    label_value_to_idx: Dict[Union[int, Tuple], int] = {}
    for class_idx, (label_value, class_name) in enumerate(id_mapping.items()):
        assert isinstance(class_name, str), "You need to format id_mapping with key for class_index, value for class_name."
        if isinstance(label_value, (int, tuple)):
            idx_to_class[class_idx] = class_name
            label_value_to_idx[label_value] = class_idx
            continue

        # Check tuple string
        assert isinstance(label_value, str) and label_value.strip().startswith("(") and label_value.strip().endswith(")"), \
            f"Unknown type for color index! Should be one of (int, tuple-style str, tuple)... but {type(label_value)}"
        label_value_tuple: Tuple[int, int, int] = as_tuple(label_value)
        idx_to_class[class_idx] = class_name
        label_value_to_idx[label_value_tuple] = class_idx

    return idx_to_class, label_value_to_idx


class SegmentationDataSampler(BaseDataSampler):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(SegmentationDataSampler, self).__init__(conf_data, train_valid_split_ratio)

    def load_data(self, split='train'):
        data_root = Path(self.conf_data.path.root)
        split_dir = self.conf_data.path[split]
        image_dir: Path = data_root / split_dir.image
        annotation_dir: Path = data_root / split_dir.label
        images: List[str] = []
        labels: List[str] = []
        images_and_targets: List[Dict[str, str]] = []
        if split in ['train', 'valid']:
            for ext in IMG_EXTENSIONS:
                images.extend([str(file) for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])
                # TODO: get paired data from regex pattern matching (conf_data.path.pattern)
                labels.extend([str(file) for file in chain(annotation_dir.glob(f'*{ext}'), annotation_dir.glob(f'*{ext.upper()}'))])

            images = sorted(images, key=lambda k: natural_key(k))
            labels = sorted(labels, key=lambda k: natural_key(k))
            images_and_targets.extend([{'image': str(image), 'label': str(label)} for image, label in zip(images, labels)])

        elif split == 'test':
            for ext in IMG_EXTENSIONS:
                images_and_targets.extend([{'image': str(file), 'label': None}
                                           for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k['image']))
        else:
            raise AssertionError(f"split should be either {['train', 'valid', 'test']}")

        return images_and_targets

    def load_samples(self):
        assert self.conf_data.path.train.image is not None
        assert isinstance(self.conf_data.id_mapping, (ListConfig, DictConfig))

        idx_to_class, label_value_to_idx = load_custom_class_map(id_mapping=self.conf_data.id_mapping)

        exists_valid = self.conf_data.path.valid.image is not None
        exists_test = self.conf_data.path.test.image is not None

        valid_samples = None
        test_samples = None

        train_samples = self.load_data(split='train')
        if exists_valid:
            valid_samples = self.load_data(split='valid')
        if exists_test:
            test_samples = self.load_data(split='test')

        if not exists_valid:
            num_train_splitted = int(len(train_samples) * self.train_valid_split_ratio)
            train_samples, valid_samples = random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                                                        generator=torch.Generator().manual_seed(42))

        return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class, 'label_value_to_idx': label_value_to_idx}

    def load_huggingface_samples(self):
        from datasets import load_dataset

        cache_dir = self.conf_data.metadata.custom_cache_dir
        root = self.conf_data.metadata.repo
        subset_name = self.conf_data.metadata.subset
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
        total_dataset = load_dataset(root, name=subset_name, cache_dir=cache_dir)

        assert isinstance(self.conf_data.id_mapping, (ListConfig, DictConfig))

        idx_to_class, label_value_to_idx = load_custom_class_map(id_mapping=self.conf_data.id_mapping)

        exists_valid = 'validation' in total_dataset
        exists_test = 'test' in total_dataset

        train_samples = total_dataset['train']
        valid_samples = None
        if exists_valid:
            valid_samples = total_dataset['validation']
        test_samples = None
        if exists_test:
            test_samples = total_dataset['test']

        if not exists_valid:
            splitted_datasets = train_samples.train_test_split(test_size=(1 - self.train_valid_split_ratio))
            train_samples = splitted_datasets['train']
            valid_samples = splitted_datasets['test']
        return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class, 'label_value_to_idx': label_value_to_idx}
