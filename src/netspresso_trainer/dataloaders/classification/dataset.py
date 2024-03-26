import csv
import random
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import random_split

from ..base import BaseDataSampler
from ..utils.constants import IMG_EXTENSIONS
from ..utils.misc import natural_key

VALID_IMG_EXTENSIONS = IMG_EXTENSIONS + tuple((x.upper() for x in IMG_EXTENSIONS))


def load_custom_class_map(id_mapping: List[str]):
    idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
    return idx_to_class


def load_class_map_with_id_mapping(labels_path: Optional[Union[str, Path]]):
    # Assume the `map_or_filename` is path for csv label file
    assert labels_path.exists(), f"Cannot locate specified class map file {labels_path}!"
    class_map_ext = labels_path.suffix.lower()
    assert class_map_ext == '.csv', f"Unsupported class map file extension ({class_map_ext})!"

    with open(labels_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        file_to_idx = {row['image_id']: int(row['class']) for row in reader}

    return file_to_idx


def is_file_dict(image_dir: Union[Path, str], file_or_dir_to_idx):
    image_dir = Path(image_dir)
    candidate_name = list(file_or_dir_to_idx.keys())[0]
    file_or_dir: Path = image_dir / candidate_name
    if file_or_dir.exists():
        return file_or_dir.is_file()

    file_candidates = list(image_dir.glob(f"{candidate_name}.*"))
    assert len(file_candidates) != 0, f"Unknown label format! Is there any something file like {file_or_dir} ?"

    return True


def classification_mix_collate_fn(original_batch, mix_transforms):
    indices = []
    images = []
    target = []
    for data_sample in original_batch:
        indices.append(data_sample[0])
        images.append(data_sample[1])
        target.append(data_sample[2])

    indices = torch.tensor(indices, dtype=torch.long)
    images = torch.stack(images, dim=0)
    target = torch.tensor(target, dtype=torch.long)

    images, target = mix_transforms(images, target)

    outputs = (indices, images, target)
    return outputs


def classification_onehot_collate_fn(original_batch, num_classes):
    indices = []
    images = []
    target = []
    for data_sample in original_batch:
        indices.append(data_sample[0])
        images.append(data_sample[1])
        target.append(data_sample[2])

    indices = torch.tensor(indices, dtype=torch.long)
    images = torch.stack(images, dim=0)
    target = torch.tensor(target, dtype=torch.long)
    if -1 not in target:
        target = F.one_hot(target, num_classes=num_classes).to(dtype=images.dtype)

    outputs = (indices, images, target)
    return outputs


class ClassficationDataSampler(BaseDataSampler):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(ClassficationDataSampler, self).__init__(conf_data, train_valid_split_ratio)

    def load_data(self, split='train'):
        data_root = Path(self.conf_data.path.root)
        split_dir = self.conf_data.path[split]
        image_dir: Path = data_root / split_dir.image
        annotation_path: Optional[Path] = data_root / split_dir.label if split_dir.label is not None else None
        images_and_targets: List[Dict[str, Optional[Union[str, int]]]] = []

        assert split in ['train', 'valid', 'test'], f"split should be either {['train', 'valid', 'test']}"
        if annotation_path is not None:
            file_to_idx = load_class_map_with_id_mapping(annotation_path)
            for ext in IMG_EXTENSIONS:
                for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}')):
                    if file.name in file_to_idx:
                        images_and_targets.append({'image': str(file), 'label': file_to_idx[file.name]})
                        continue
                    logger.debug(f"Found file without label: {file}")
        else:
            if split in ['train', 'valid']:
                raise ValueError("For train and valid split, label path must be provided!")
            for ext in VALID_IMG_EXTENSIONS:
                images_and_targets.extend([{'image': str(file), 'label': None} for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])

        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k['image']))
        return images_and_targets

    def load_samples(self):
        assert self.conf_data.id_mapping is not None
        id_mapping = list(self.conf_data.id_mapping)
        idx_to_class = load_custom_class_map(id_mapping=id_mapping)

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

        if not exists_valid:
            num_train_splitted = int(len(train_samples) * self.train_valid_split_ratio)
            train_samples, valid_samples = \
                random_split(train_samples, [num_train_splitted, len(train_samples) - num_train_splitted],
                                generator=torch.Generator().manual_seed(42))

        return train_samples, valid_samples, test_samples, {'idx_to_class': idx_to_class}

    def load_huggingface_samples(self):
        from datasets import ClassLabel, load_dataset

        cache_dir = self.conf_data.metadata.custom_cache_dir
        root = self.conf_data.metadata.repo
        subset_name = self.conf_data.metadata.subset
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
        total_dataset = load_dataset(root, name=subset_name, cache_dir=cache_dir)

        label_feature_name = self.conf_data.metadata.features.label
        # Assumed hugging face dataset always has training split
        label_feature = total_dataset['train'].features[label_feature_name]
        if isinstance(label_feature, ClassLabel):
            labels: List[str] = label_feature.names
        else:
            labels = list({sample[label_feature_name] for sample in total_dataset['train']})

        if isinstance(labels[0], int):
            # TODO: find class_map <-> idx and apply it (ex. using id_mapping)
            idx_to_class: Dict[int, int] = {k: k for k in labels}
        elif isinstance(labels[0], str):
            idx_to_class: Dict[int, str] = dict(enumerate(labels))

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
