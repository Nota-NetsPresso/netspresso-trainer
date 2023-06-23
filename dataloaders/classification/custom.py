import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import PIL.Image as Image
import torch
from torch.utils.data import random_split
from omegaconf import DictConfig

from dataloaders.utils.parsers import create_parser
from dataloaders.base import BaseCustomDataset, BaseHFDataset
from dataloaders.utils.constants import IMG_EXTENSIONS
from dataloaders.utils.misc import natural_key
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

_ERROR_RETRY = 50

class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            idx_to_class,
            split,
            samples,
            transform=None,
    ):
        root = args.data.path.root
        super(ClassificationCustomDataset, self).__init__(
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
        img = self.samples[index]['image']
        target = self.samples[index]['label'] if 'label' in self.samples[index] else None
        img = Image.open(img).convert('RGB')
        
        if self.transform is not None:
            out = self.transform(args_augment=self.args.augment, img_size=self.args.training.img_size)(img)
        
        if target is None:
            target = -1  # To be ignored at cross-entropy loss
        return out['image'], target


class ClassificationHFDataset(BaseHFDataset):

    def __init__(
            self,
            args,
            split,
            transform=None,
            target_transform=None,
    ):
        root = args.data.metadata.repo
        super(ClassificationHFDataset, self).__init__(
            args,
            root,
            split
        )
        # Make sure that you additionally install `requirements-data.txt`

        self.transform = transform
        
        subset_name = args.data.metadata.subset
        cache_dir = args.data.metadata.custom_cache_dir
        total_dataset = self._load_dataset(root, subset_name=subset_name, cache_dir=cache_dir)

        if self._split in ['train', 'training']:
            _dataset = total_dataset['train']
        elif self._split in ['val', 'valid', 'validation']:
            _dataset = total_dataset['validation']
        else:
            raise NotImplementedError
            
        image_key = args.data.metadata.features.image
        label_key = args.data.metadata.features.label
            
        self._class_map: Dict[int, str] = {class_idx: str(class_name)
                                      for class_idx, class_name
                                      in enumerate(sorted(set([x[label_key] for x in _dataset])))}
        self._class_name_as_index: Dict[str, int] = {str(v): k for k, v in self._class_map.items()}
            
        self._num_classes = len(self._class_map)
        
        self.parser: Dict[Tuple[Image.Image, int]] = [(x[image_key], self._class_name_as_index[str(x[label_key])]) for x in _dataset]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self._class_map

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, index):
        img, target = self.parser[index]
        if self.transform is not None:
            out = self.transform(args_augment=self.args.augment, img_size=self.args.training.img_size)(img)
        if target is None:
            target = -1
        return out['image'], target
    
def load_custom_class_map(root, map_or_filename=None):

    dir_list = [x.name for x in Path(root).iterdir() if x.is_dir()]

    if map_or_filename is None:  # class_name == dir_name
        dir_to_idx = {v.strip(): k for k, v in enumerate(dir_list)}
        idx_to_class = {k: v.strip() for k, v in enumerate(dir_list)}

    else:
        if isinstance(map_or_filename, dict):
            assert dict, "class_map dict must be non-empty"
            dir_to_idx = {v.strip(): k for k, v in enumerate(map_or_filename.keys())}
            idx_to_class = {k: v.strip() for k, v in enumerate(map_or_filename.values())}
            return dir_to_idx, idx_to_class

        class_map_path = Path(map_or_filename)

        if not class_map_path.exists():
            class_map_path = Path(root) / class_map_path
            assert class_map_path.exists(), f"Cannot locate specified class map file {map_or_filename}!"

        class_map_ext = class_map_path.suffix.lower()
        assert class_map_ext == '.txt', f"Unsupported class map file extension ({class_map_ext})!"

        with open(class_map_path) as f:
            map_data = [x.strip().split(' ') for x in f.readlines()]
            dir_list_from_map = [x[0] for x in map_data]
            assert set(dir_list).issubset(set(dir_list_from_map)), \
                f"Found unknown directory in ({root}) whose class is not defined: {set(dir_list).difference(set(dir_list_from_map))}"
            class_list_from_map = [' '.join(x[1:]) for x in map_data]
            dir_to_idx = {v.strip(): k for k, v in enumerate(dir_list_from_map)}
            idx_to_class = {k: v.strip() for k, v in enumerate(class_list_from_map)}

    return dir_to_idx, idx_to_class

    
def load_data(args_data: DictConfig, dir_to_idx, split='train'):
    data_root = Path(args_data.path.root)
    split_dir = args_data.path[split]
    image_dir = data_root / split_dir.image
    
    images_and_targets = []
    for dir_name, dir_idx in dir_to_idx.items():
        _dir = Path(image_dir) / dir_name
        for ext in IMG_EXTENSIONS:
            images_and_targets.extend([{'image': str(file), 'label': dir_idx} for file in _dir.glob(f'*{ext}')])
            images_and_targets.extend([{'image': str(file), 'label': dir_idx} for file in _dir.glob(f'*{ext.upper()}')])

    images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k['image']))

    return images_and_targets


def load_samples(args_data):
    assert args_data.path.train.image is not None
    train_dir = Path(args_data.path.root) / args_data.path.train.image
    class_map: Optional[dict] = dict(args_data.id_mapping) if args_data.id_mapping is not None else None
    dir_to_idx, idx_to_class = load_custom_class_map(train_dir, class_map)
    
    exists_valid = args_data.path.valid.image is not None
    exists_test = args_data.path.test.image is not None
    
    valid_samples = None
    test_samples = None
    
    train_samples = load_data(args_data, dir_to_idx, split='train')
    if exists_valid:
        valid_samples = load_data(args_data, dir_to_idx, split='valid')
    if exists_test:
        test_samples = load_data(args_data, dir_to_idx, split='test')

    if not exists_valid and not exists_test:
        train_samples, valid_samples = \
            random_split(train_samples, [0.9, 0.1],
                            generator=torch.Generator().manual_seed(42))
    
    return train_samples, valid_samples, test_samples, \
        {'dir_to_idx': dir_to_idx, 'idx_to_class': idx_to_class}

def create_classification_dataset(args, transform, target_transform=None):
    data_format = args.data.format
    if data_format == 'local':
        # TODO: Load train/valid/test data samples
        train_samples, valid_samples, test_samples, misc = load_samples(args.data)
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        
        train_dataset = ClassificationCustomDataset(
            args, idx_to_class=idx_to_class, split='train',
            samples=train_samples, transform=transform
        )
        
        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = ClassificationCustomDataset(
                args, idx_to_class=idx_to_class, split='valid',
                samples=valid_samples, transform=target_transform
            )
        
        test_dataset = None
        if test_samples is not None:
            test_dataset = ClassificationCustomDataset(
                args, idx_to_class=idx_to_class, split='test',
                samples=test_samples, transform=target_transform
            )
        
        return train_dataset, valid_dataset, test_dataset
    elif data_format == 'huggingface':
        return ClassificationHFDataset(args, split='train',
                                       transform=transform, target_transform=target_transform)
    else:
        raise AssertionError(f"No such data format named {data_format}!")