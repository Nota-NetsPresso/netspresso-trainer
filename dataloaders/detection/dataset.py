import os
from pathlib import Path
import json
from typing import Optional, Union, Tuple, List, Dict
from itertools import chain

import PIL.Image as Image
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import random_split
import torch

from dataloaders.detection.local import DetectionCustomDataset
# from dataloaders.detection.huggingface import DetectionHFDataset
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


def get_label(label_file: Path):
    target = Path(label_file).read_text()
    
    try:
        target_array = np.array([list(map(float, box.split(' '))) for box in target.split('\n') if box.strip()])
    except ValueError as e:
        print(target)
        raise e
        
    label, boxes = target_array[:, 0], target_array[:, 1:]
    label = label[..., np.newaxis]
    return label, boxes

def load_custom_class_map(id_mapping: List[str]):
    idx_to_class: Dict[int, str] = {k: v for k, v in enumerate(id_mapping)}
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
            for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}')):
                ann_path_maybe = annotation_dir / file.with_suffix('.txt').name
                if not ann_path_maybe.exists():
                    continue
                images.append(str(file))
                labels.append(str(ann_path_maybe))
            # TODO: get paired data from regex pattern matching (args.data.path.pattern)

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

def load_samples_local(args_data):
    assert args_data.path.train.image is not None
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


def create_detection_dataset(args, transform, target_transform=None):
    data_format = args.data.format
    if data_format == 'local':
        train_samples, valid_samples, test_samples, misc = load_samples_local(args.data)
        idx_to_class = misc['idx_to_class'] if 'idx_to_class' in misc else None
        
        train_dataset = DetectionCustomDataset(
            args, idx_to_class=idx_to_class, split='train',
            samples=train_samples, transform=transform
        )
        
        valid_dataset = None
        if valid_samples is not None:
            valid_dataset = DetectionCustomDataset(
                args, idx_to_class=idx_to_class, split='valid',
                samples=valid_samples, transform=target_transform
            )
        
        test_dataset = None
        if test_samples is not None:
            test_dataset = DetectionCustomDataset(
                args, idx_to_class=idx_to_class, split='test',
                samples=test_samples, transform=target_transform
            )
        
        return train_dataset, valid_dataset, test_dataset 
    elif data_format == 'huggingface':
        raise NotImplementedError(f"Currently, detection training with Hugging Face dataset is not supported.")
    else:
        raise AssertionError(f"No such data format named {data_format}!")