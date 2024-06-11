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

import json
import os
from functools import partial
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import PIL.Image as Image
import torch.distributed as dist
from loguru import logger
from omegaconf import ListConfig

from .base import BaseCustomDataset, BaseHFDataset, BaseSampleLoader
from .utils.constants import IMG_EXTENSIONS
from .utils.misc import load_classification_class_map, natural_key

VALID_IMG_EXTENSIONS = IMG_EXTENSIONS + tuple((x.upper() for x in IMG_EXTENSIONS))


class ClassficationSampleLoader(BaseSampleLoader):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(ClassficationSampleLoader, self).__init__(conf_data, train_valid_split_ratio)

    def load_data(self, split='train'):
        data_root = Path(self.conf_data.path.root)
        split_dir = self.conf_data.path[split]
        image_dir: Path = data_root / split_dir.image
        annotation_path: Optional[Path] = data_root / split_dir.label if split_dir.label is not None else None
        images_and_targets: List[Dict[str, Optional[Union[str, int]]]] = []

        assert split in ['train', 'valid', 'test'], f"split should be either {['train', 'valid', 'test']}"
        if annotation_path is not None:
            file_to_idx = load_classification_class_map(annotation_path)
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

    def load_id_mapping(self):
        root_path = Path(self.conf_data.path.root)

        if isinstance(self.conf_data.id_mapping, ListConfig):
            return list(self.conf_data.id_mapping)

        elif isinstance(self.conf_data.id_mapping, str):
            id_mapping_path = root_path / self.conf_data.id_mapping
            if not os.path.isfile(id_mapping_path):
                FileNotFoundError(f"File not found: {id_mapping_path}")

            with open(id_mapping_path, 'r') as f:
                id_mapping = json.load(f)
            return id_mapping

        else:
            raise ValueError(f"Unsupported id_mapping value {self.conf_data.id_mapping}")

    def load_class_map(self, id_mapping):
        idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
        return {'idx_to_class': idx_to_class}

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


class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, **kwargs):
        super(ClassificationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, **kwargs
        )

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(str(samples[i]['image'])).convert('RGB')
            return i, image

        num_threads = 8 # TODO: Compute appropriate num_threads
        load_imgs = ThreadPool(num_threads).imap(
            partial(_load, samples=self.samples),
            sampler
        )
        for i, image in load_imgs:
            self.samples[i]['image'] = image

        self.cache = True

    def __getitem__(self, index):
        img = self.samples[index]['image']
        target = self.samples[index]['label']
        if not self.cache:
            img = Image.open(img).convert('RGB')

        if self.transform is not None:
            out = self.transform(img)

        if target is None:
            target = -1  # To be ignored at cross-entropy loss
        return index, out['image'], target


class ClassificationHFDataset(BaseHFDataset):

    def __init__(
            self,
            conf_data,
            conf_augmentation,
            model_name,
            idx_to_class,
            split,
            huggingface_dataset,
            transform=None,
            **kwargs
    ):
        root = conf_data.metadata.repo
        super(ClassificationHFDataset, self).__init__(
            conf_data,
            conf_augmentation,
            model_name,
            root,
            split,
            transform,
        )
        # Make sure that you additionally install `requirements-data.txt`

        self.samples = huggingface_dataset
        self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}

        self.image_feature_name = conf_data.metadata.features.image
        self.label_feature_name = conf_data.metadata.features.label

    @property
    def num_classes(self):
        return len(self.idx_to_class)

    @property
    def class_map(self):
        return self.idx_to_class

    def __len__(self):
        return self.samples.num_rows

    def __getitem__(self, index):
        img: Image.Image = self.samples[index][self.image_feature_name]
        target: Union[int, str] = self.samples[index][self.label_feature_name] if self.label_feature_name in self.samples[index] else None
        if isinstance(target, str):
            target: int = self.class_to_idx[target]

        if self.transform is not None:
            out = self.transform(img)
        if target is None:
            target = -1
        return index, out['image'], target
