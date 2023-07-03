import os
import csv
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
from collections import Counter
from itertools import chain

import torch
from torch.utils.data import random_split
from omegaconf import DictConfig

from dataloaders.classification.local import ClassificationCustomDataset
from dataloaders.classification.huggingface import ClassificationHFDataset
from dataloaders.utils.constants import IMG_EXTENSIONS
from dataloaders.utils.misc import natural_key
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

TRAIN_VALID_SPLIT_RATIO = 0.9
_ERROR_RETRY = 50


    
def load_custom_class_map(root_dir, train_dir,
                          map_or_filename: Optional[Union[str, Path]]=None,
                          id_mapping: Optional[Dict[str, str]]=None):

    if map_or_filename is None:  # may be labeled with directory
        # dir -> 
        dir_list = [x.name for x in Path(train_dir).iterdir() if x.is_dir()]
        dir_to_class = id_mapping if id_mapping is not None else {k: k for k in dir_list}  # id_mapping or identity
        
        class_list = [dir_to_class[dir] for dir in dir_list]
        class_list = sorted(class_list, key=lambda k: natural_key(k))
        _class_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(class_list)}
        idx_to_class = {v: k for k, v in _class_to_idx.items()}
        
        file_or_dir_to_idx = {dir: _class_to_idx[dir_to_class[dir]] for dir in dir_list}  # dir -> idx
        return file_or_dir_to_idx, idx_to_class

    # Assume the `map_or_filename` is path for csv label file
    class_map_path = Path(root_dir) / map_or_filename
    assert class_map_path.exists(), f"Cannot locate specified class map file {class_map_path}!"

    class_map_ext = class_map_path.suffix.lower()
    assert class_map_ext == '.csv', f"Unsupported class map file extension ({class_map_ext})!"

    with open(class_map_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        file_class_list = [{column: str(row[column]).strip() for column in ['image_id', 'class']}
                           for row in reader]
    
    class_stats = Counter([x['class'] for x in file_class_list])
    
    _class_to_idx = {class_name: class_idx
                    for class_idx, class_name in enumerate(sorted(class_stats, key=lambda k: natural_key(k)))}
    idx_to_class = {v: k for k, v in _class_to_idx.items()}

    file_or_dir_to_idx = {elem['image_id']: _class_to_idx[elem['class']] for elem in file_class_list}  # file -> idx

    return file_or_dir_to_idx, idx_to_class

def is_file_dict(image_dir: Union[Path, str], file_or_dir_to_idx):
    image_dir = Path(image_dir)
    candidate_name = list(file_or_dir_to_idx.keys())[0]
    file_or_dir: Path = image_dir / candidate_name
    if file_or_dir.exists():
        return file_or_dir.is_file()
    
    file_candidates = [x for x in image_dir.glob(f"{candidate_name}.*")]
    assert len(file_candidates) != 0, f"Unknown label format! Is there any something file like {file_or_dir} ?"
    
    return True
    
    
def load_data(args_data: DictConfig, file_or_dir_to_idx, split='train'):
    data_root = Path(args_data.path.root)
    split_dir = args_data.path[split]
    image_dir: Path = data_root / split_dir.image
    
    images_and_targets: List[Dict[str, Optional[Union[str, int]]]] = []
    if split in ['train', 'valid']:
        
        if is_file_dict(image_dir, file_or_dir_to_idx):
            file_to_idx = file_or_dir_to_idx
            for ext in IMG_EXTENSIONS:
                for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}')):
                    if file.name in file_to_idx:
                        images_and_targets.append({'image': str(file), 'label': file_to_idx[file.name]})
                        continue
                    if file.stem in file_to_idx:
                        images_and_targets.append({'image': str(file), 'label': file_to_idx[file.stem]})
                        continue
                    logger.debug(f"Found file wihtout label: {file}")
        
        else:
            dir_to_idx = file_or_dir_to_idx
            for dir_name, dir_idx in dir_to_idx.items():
                _dir = Path(image_dir) / dir_name
                for ext in IMG_EXTENSIONS:
                    images_and_targets.extend([{'image': str(file), 'label': dir_idx}
                                               for file in chain(_dir.glob(f'*{ext}'), _dir.glob(f'*{ext.upper()}'))])

        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k['image']))
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
    train_dir = root_dir / args_data.path.train.image
    id_mapping: Optional[dict] = dict(args_data.id_mapping) if args_data.id_mapping is not None else None
    file_or_dir_to_idx, idx_to_class = load_custom_class_map(root_dir, train_dir, map_or_filename=args_data.path.train.label, id_mapping=id_mapping)
    
    exists_valid = args_data.path.valid.image is not None
    exists_test = args_data.path.test.image is not None
    
    valid_samples = None
    test_samples = None
    
    train_samples = load_data(args_data, file_or_dir_to_idx, split='train')
    if exists_valid:
        valid_samples = load_data(args_data, file_or_dir_to_idx, split='valid')
    if exists_test:
        test_samples = load_data(args_data, file_or_dir_to_idx, split='test')

    if not exists_valid:
        num_train_splitted = int(len(train_samples) * TRAIN_VALID_SPLIT_RATIO) 
        train_samples, valid_samples = \
            random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                            generator=torch.Generator().manual_seed(42))
    
    return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class}

def create_classification_dataset(args, transform, target_transform=None):
    data_format = args.data.format
    if data_format == 'local':
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