import json
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from ..base import BaseDataSampler
from ..utils.constants import IMG_EXTENSIONS
from ..utils.misc import natural_key


def load_custom_class_map(id_mapping: List[str]):
    idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
    return idx_to_class

def detection_collate_fn(original_batch):
    pixel_values = []
    bbox = []
    label = []
    org_shape = []
    for data_sample in original_batch:
        if 'pixel_values' in data_sample:
            pixel_values.append(data_sample['pixel_values'])
        if 'bbox' in data_sample:
            bbox.append(data_sample['bbox'])
        if 'label' in data_sample:
            label.append(data_sample['label'])
        if 'org_shape' in data_sample:
            org_shape.append(data_sample['org_shape'])
    outputs = {}
    if len(pixel_values) != 0:
        pixel_values = torch.stack(pixel_values, dim=0)
        outputs.update({'pixel_values': pixel_values})
    if len(bbox) != 0:
        outputs.update({'bbox': bbox})
    if len(label) != 0:
        outputs.update({'label': label})
    if len(org_shape) != 0:
        outputs.update({'org_shape': org_shape})

    return outputs

class DetectionDataSampler(BaseDataSampler):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(DetectionDataSampler, self).__init__(conf_data, train_valid_split_ratio)

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
                for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}')):
                    ann_path_maybe = annotation_dir / file.with_suffix('.txt').name
                    if not ann_path_maybe.exists():
                        continue
                    images.append(str(file))
                    labels.append(str(ann_path_maybe))
                # TODO: get paired data from regex pattern matching (self.conf_data.path.pattern)

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
        raise NotImplementedError
