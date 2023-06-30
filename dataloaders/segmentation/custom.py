import os
from pathlib import Path
import logging
import json
from typing import Optional, Union, Tuple, List, Dict
from collections import Counter
from itertools import chain

import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import random_split
from omegaconf import DictConfig

from dataloaders.base import BaseCustomDataset, BaseHFDataset
from dataloaders.segmentation.transforms import generate_edge, reduce_label
from dataloaders.utils.constants import IMG_EXTENSIONS
from dataloaders.utils.misc import natural_key
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

TRAIN_VALID_SPLIT_RATIO = 0.9

def exist_name(candidate, folder_iterable):
    try:
        return list(filter(lambda x: candidate[0] in x, folder_iterable))[0]
    except:
        return list(filter(lambda x: candidate[1] in x, folder_iterable))[0]


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


class SegmentationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            idx_to_class,
            split,
            samples,
            transform=None,
    ):
        root = args.data.path.root
        super(SegmentationCustomDataset, self).__init__(
            args,
            root,
            split
        )
        
        self.transform = transform

        self.samples = samples
        self.idx_to_class = idx_to_class
        self._num_classes = len(self.idx_to_class)


    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self.idx_to_class
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = Path(self.samples[index]['image'])
        ann_path = Path(self.samples[index]['label']) if 'label' in self.samples[index] else None
        img = Image.open(img_path).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if ann_path is None:
            out = self.transform(self.args.augment)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        label = Image.open(ann_path).convert('L')
        if self.args.augment.reduce_zero_label:
            label = reduce_label(np.array(label))

        if self.args.train.architecture.full == 'pidnet':
            edge = generate_edge(np.array(label))
            out = self.transform(self.args.augment)(image=img, mask=label, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_path.name})
        else:
            out = self.transform(self.args.augment)(image=img, mask=label)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs

class SegmentationHFDataset(BaseHFDataset):

    def __init__(
            self,
            args,
            split,
            transform=None,
            target_transform=None,
            load_bytes=False,
    ):
        root = args.data.metadata.repo
        super(SegmentationHFDataset, self).__init__(
            args,
            root,
            split
        )
        
        raise NotImplementedError

        if self._split in ['train', 'training', 'val', 'valid', 'test']:  # for training and test (= evaluation) phase
            self.image_dir = Path(self._root) / args.data.path.train.image
            self.annotation_dir = Path(self._root) / args.data.path.train.mask

            self.id2label = args.data.id_mapping

            self.img_name = list(sorted([path for path in self.image_dir.iterdir()]))
            self.ann_name = list(sorted([path for path in self.annotation_dir.iterdir()]))
            # TODO: get paired data from regex pattern matching (args.data.path.pattern)

            assert len(self.img_name) == len(self.ann_name), "There must be as many images as there are segmentation maps"

        else:  # self._split in ['infer', 'inference']
            raise NotImplementedError
            try:  # a folder with multiple images
                self.img_name = list(sorted([path for path in Path(self.data_dir).iterdir()]))
            except:  # single image
                raise AssertionError
                # TODO: check the case for single image
                self.file_name = [self.data_dir.split('/')[-1]]
                self.img_name = [self.data_dir]

        self.transform = transform

    def __len__(self):
        raise NotImplementedError
        return len(self.img_name)

    @property
    def num_classes(self):
        raise NotImplementedError
        return len(self.id2label)

    @property
    def class_map(self):
        raise NotImplementedError
        return self.id2label

    def __getitem__(self, index):
        raise NotImplementedError
        img_path = self.img_name[index]
        ann_path = self.ann_name[index]
        img = Image.open(str(img_path)).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if self._split in ['infer', 'inference']:
            out = self.transform(self.args.augment)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        label = Image.open(str(ann_path)).convert('L')
        if self.args.augment.reduce_zero_label:
            label = reduce_label(np.array(label))

        if self.args.train.architecture.full == 'pidnet':
            edge = generate_edge(np.array(label))
            out = self.transform(self.args.augment)(image=img, mask=label, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_path.name})
        else:
            out = self.transform(self.args.augment)(image=img, mask=label)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
    
def load_custom_class_map(id_mapping: List[str]):
    idx_to_class: Dict[int, str] = {k: v for k, v in enumerate(id_mapping)}
    return idx_to_class

def load_data(args_data: DictConfig, split='train'):
    data_root = Path(args_data.path.root)
    split_dir = args_data.path[split]
    image_dir: Path = data_root / split_dir.image
    annotation_dir: Path = data_root / split_dir.label
    images: List[str] = []
    labels: List[str] = []
    images_and_targets: List[Dict[str, str]] = []
    if split in ['train', 'valid']:
        for ext in IMG_EXTENSIONS:
            images.extend([str(file) for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])
            # TODO: get paired data from regex pattern matching (args.data.path.pattern)
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


def load_samples(args_data):
    assert args_data.path.train.image is not None
    root_dir = Path(args_data.path.root)
    assert args_data.id_mapping is not None
    id_mapping: Optional[list] = list(args_data.id_mapping)
    idx_to_class = load_custom_class_map(id_mapping=id_mapping)
    
    exists_valid = args_data.path.valid.image is not None
    exists_test = args_data.path.test.image is not None
    
    valid_samples = None
    test_samples = None
    
    train_samples = load_data(args_data, split='train')
    if exists_valid:
        valid_samples = load_data(args_data, split='valid')
    if exists_test:
        test_samples = load_data(args_data, split='test')

    if not exists_valid:
        num_train_splitted = int(len(train_samples) * TRAIN_VALID_SPLIT_RATIO) 
        train_samples, valid_samples = \
            random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                            generator=torch.Generator().manual_seed(42))
    
    return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class}


def create_segmentation_dataset(args, transform, target_transform=None):
    data_format = args.data.format
    if data_format == 'local':
        # TODO: Load train/valid/test data samples
        train_samples, valid_samples, test_samples, misc = load_samples(args.data)
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        
        train_dataset = SegmentationCustomDataset(
            args, idx_to_class=idx_to_class, split='train',
            samples=train_samples, transform=transform
        )
        
        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = SegmentationCustomDataset(
                args, idx_to_class=idx_to_class, split='valid',
                samples=valid_samples, transform=target_transform
            )
        
        test_dataset = None
        if test_samples is not None:
            test_dataset = SegmentationCustomDataset(
                args, idx_to_class=idx_to_class, split='test',
                samples=test_samples, transform=target_transform
            )
        
        return train_dataset, valid_dataset, test_dataset        
    elif data_format == 'huggingface':
        return SegmentationHFDataset(args, transform=transform, target_transform=target_transform)
    else:
        raise AssertionError(f"No such data format named {data_format}!")