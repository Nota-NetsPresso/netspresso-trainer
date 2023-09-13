import json
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from ..base import BaseDataSampler
from ..utils.constants import IMG_EXTENSIONS
from ..utils.misc import natural_key


def load_custom_class_map(id_mapping: List[str]):
    idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
    return idx_to_class

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
        assert self.conf_data.id_mapping is not None
        id_mapping: Optional[list] = list(self.conf_data.id_mapping)
        idx_to_class = load_custom_class_map(id_mapping=id_mapping)
        
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
            train_samples, valid_samples = \
                random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                                generator=torch.Generator().manual_seed(42))
        
        return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class}
    
    def load_huggingface_samples(self):
        from datasets import load_dataset
        
        cache_dir = Path(self.conf_data.metadata.custom_cache_dir)
        root = self.conf_data.metadata.repo
        subset_name = self.conf_data.metadata.subset
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
        total_dataset = load_dataset(root, name=subset_name, cache_dir=cache_dir)
        
        assert self.conf_data.metadata.id_mapping is not None
        id_mapping: Optional[list] = list(self.conf_data.metadata.id_mapping)
        idx_to_class = load_custom_class_map(id_mapping=id_mapping)
        
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
        return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class}