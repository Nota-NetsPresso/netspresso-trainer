import os
import csv
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
from collections import Counter
from itertools import chain

import torch
from torch.utils.data import random_split
from omegaconf import DictConfig

from ..base import BaseDataSampler
from ..utils.constants import IMG_EXTENSIONS
from ..utils.misc import natural_key
from ...utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))
    
def load_class_map_with_id_mapping(root_dir, train_dir,
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

class ClassficationDataSampler(BaseDataSampler):
    def __init__(self, args_data, train_valid_split_ratio):
        super(ClassficationDataSampler, self).__init__(args_data, train_valid_split_ratio)
    
    def load_data(self, file_or_dir_to_idx, split='train'):
        data_root = Path(self.args_data.path.root)
        split_dir = self.args_data.path[split]
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
        
    def load_samples(self):
        assert self.args_data.path.train.image is not None
        root_dir = Path(self.args_data.path.root)
        train_dir = root_dir / self.args_data.path.train.image
        id_mapping: Optional[dict] = dict(self.args_data.id_mapping) if self.args_data.id_mapping is not None else None
        file_or_dir_to_idx, idx_to_class = load_class_map_with_id_mapping(root_dir, train_dir, map_or_filename=self.args_data.path.train.label, id_mapping=id_mapping)
        
        exists_valid = self.args_data.path.valid.image is not None
        exists_test = self.args_data.path.test.image is not None
        
        valid_samples = None
        test_samples = None
        
        train_samples = self.load_data(file_or_dir_to_idx, split='train')
        if exists_valid:
            valid_samples = self.load_data(file_or_dir_to_idx, split='valid')
        if exists_test:
            test_samples = self.load_data(file_or_dir_to_idx, split='test')

        if not exists_valid:
            num_train_splitted = int(len(train_samples) * self.train_valid_split_ratio) 
            train_samples, valid_samples = \
                random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                                generator=torch.Generator().manual_seed(42))
        
        return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class}
    
    def load_huggingface_samples(self):
        from datasets import load_dataset
        from datasets import ClassLabel
        
        cache_dir = Path(self.args_data.metadata.custom_cache_dir)
        root = self.args_data.metadata.repo
        subset_name = self.args_data.metadata.subset
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
        total_dataset = load_dataset(root, name=subset_name, cache_dir=cache_dir)
        
        label_feature_name = self.args_data.metadata.features.label
        label_feature = total_dataset['train'].features[label_feature_name]
        if isinstance(label_feature, ClassLabel):
            labels: List[str] = label_feature.names
        else:
            labels = list({sample[label_feature_name] for sample in total_dataset['train']})
            
        if isinstance(labels[0], int):
            # TODO: find class_map <-> idx and apply it (ex. using id_mapping)
            idx_to_class: Dict[int, int] = {k: k for k in labels}
        elif isinstance(labels[0], str):
            idx_to_class: Dict[int, str] = {k: v for k, v in enumerate(labels)}
        
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