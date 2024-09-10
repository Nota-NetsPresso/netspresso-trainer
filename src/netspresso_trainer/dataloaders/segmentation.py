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
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig, ListConfig

from .augmentation.transforms import generate_edge
from .base import BaseCustomDataset, BaseHFDataset, BaseSampleLoader
from .utils.constants import IMG_EXTENSIONS
from .utils.misc import as_tuple, natural_key


class SegmentationSampleLoader(BaseSampleLoader):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(SegmentationSampleLoader, self).__init__(conf_data, train_valid_split_ratio)

    def load_data(self, split='train'):
        assert split in ['train', 'valid', 'test'], f"split should be either {['train', 'valid', 'test']}."
        data_root = Path(self.conf_data.path.root)
        split_dir = self.conf_data.path[split]
        image_dir: Path = data_root / split_dir.image
        annotation_dir: Optional[Path] = data_root / split_dir.label if split_dir.label is not None else None
        images: List[str] = []
        labels: List[str] = []
        images_and_targets: List[Dict[str, str]] = []
        if annotation_dir is not None:
            for ext in IMG_EXTENSIONS:
                images.extend([str(file) for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])
                # TODO: get paired data from regex pattern matching (conf_data.path.pattern)
                labels.extend([str(file) for file in chain(annotation_dir.glob(f'*{ext}'), annotation_dir.glob(f'*{ext.upper()}'))])

            images = sorted(images, key=lambda k: natural_key(k))
            labels = sorted(labels, key=lambda k: natural_key(k))
            images_and_targets.extend([{'image': str(image), 'label': str(label)} for image, label in zip(images, labels)])

        else:
            for ext in IMG_EXTENSIONS:
                images_and_targets.extend([{'image': str(file), 'label': None}
                                           for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k['image']))

        return images_and_targets

    def load_id_mapping(self):
        root_path = Path(self.conf_data.path.root)

        if isinstance(self.conf_data.id_mapping, DictConfig):
            return dict(self.conf_data.id_mapping)
        elif isinstance(self.conf_data.id_mapping, ListConfig):
            return dict(enumerate(self.conf_data.id_mapping))
        elif isinstance(self.conf_data.id_mapping, str):
            id_mapping_path = root_path / self.conf_data.id_mapping
            if not os.path.exists(id_mapping_path):
                raise FileNotFoundError(f"File not found: {id_mapping_path}")

            with open(id_mapping_path, 'r') as f:
                id_mapping = json.load(f)
            if isinstance(id_mapping, list):
                id_mapping = dict(enumerate(id_mapping))

            return id_mapping
        else:
            raise ValueError(f"Unknown type for id_mapping! {type(self.conf_data.id_mapping)}")

    def load_class_map(self, id_mapping):
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

        return {'idx_to_class': idx_to_class, 'label_value_to_idx': label_value_to_idx}

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
        if isinstance(self.conf_data.id_mapping, ListConfig):
            self.conf_data.id_mapping = dict(enumerate(self.conf_data.id_mapping))

        misc = self.load_class_map(id_mapping=self.conf_data.id_mapping)

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
        return train_samples, valid_samples, test_samples, misc


class SegmentationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, **kwargs):
        super(SegmentationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, **kwargs
        )
        assert "label_value_to_idx" in kwargs
        self.label_value_to_idx = kwargs["label_value_to_idx"]

        self.label_image_mode: Literal['RGB', 'L', 'P'] = str(conf_data.label_image_mode).upper() \
            if conf_data.label_image_mode is not None else 'L'

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(Path(samples[i]['image'])).convert('RGB')
            label = self.samples[i]['label']
            if label is not None:
                label = Image.open(Path(label)).convert(self.label_image_mode)
            return i, image, label

        num_threads = 8 # TODO: Compute appropriate num_threads
        load_imgs = ThreadPool(num_threads).imap(
            partial(_load, samples=self.samples),
            sampler
        )
        for i, image, label in load_imgs:
            self.samples[i]['image'] = image
            self.samples[i]['label'] = label

        self.cache = True

    def __getitem__(self, index):
        if self.cache:
            img = self.samples[index]['image']
            label = self.samples[index]['label']
        else:
            img_path = self.samples[index]['image']
            ann_path = self.samples[index]['label']
            img = Image.open(Path(img_path)).convert('RGB')
            label = Image.open(Path(ann_path)) if ann_path is not None else None

        w, h = img.size

        outputs = {}
        outputs.update({'indices': index})
        if label is None:
            out = self.transform(image=img)
            outputs.update({'pixel_values': out['image'], 'org_shape': (h, w)})
            return outputs

        if self.label_image_mode == 'L':
            mask = label
        else: # RGB, P
            label = label.convert('RGB')
            label = np.array(label)
            mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8) + 255 # Set undefined as 255
            for label_value in self.label_value_to_idx:
                class_mask = (label == np.array(label_value)).all(axis=-1)
                mask[class_mask] = self.label_value_to_idx[label_value]
            mask = Image.fromarray(mask, mode='L')

        if 'pidnet' in self.model_name:
            edge = generate_edge(np.array(mask))
            out = self.transform(image=img, mask=mask, edge=edge, dataset=self)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float()})
        else:
            out = self.transform(image=img, mask=mask, dataset=self)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask']})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs


class SegmentationHFDataset(BaseHFDataset):

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
        super(SegmentationHFDataset, self).__init__(
            conf_data,
            conf_augmentation,
            model_name,
            root,
            split,
            transform,
        )

        self.idx_to_class = idx_to_class
        self.samples = huggingface_dataset

        assert "label_value_to_idx" in kwargs
        self.label_value_to_idx = kwargs["label_value_to_idx"]

        self.label_image_mode: Literal['RGB', 'L', 'P'] = str(conf_data.label_image_mode).upper() \
            if conf_data.label_image_mode is not None else 'L'

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

        img_name = f"{index:06d}"
        img: Image.Image = self.samples[index][self.image_feature_name]
        label: Image.Image = self.samples[index][self.label_feature_name] if self.label_feature_name in self.samples[index] else None

        if self.label_image_mode == 'L':
            mask = label
        else: # RGB, P
            label = label.convert('RGB')
            label = np.array(label)
            mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8) + 255 # Set undefined as 255
            for label_value in self.label_value_to_idx:
                class_mask = (label == np.array(label_value)).all(axis=-1)
                mask[class_mask] = self.label_value_to_idx[label_value]
            mask = Image.fromarray(mask, mode='L')

        w, h = img.size

        if label is None:
            out = self.transform(image=img)
            return {'pixel_values': out['image'], 'name': img_name, 'org_shape': (h, w)}

        outputs = {}

        if 'pidnet' in self.model_name:
            edge = generate_edge(np.array(label))
            out = self.transform(image=img, mask=mask, edge=edge, dataset=self)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_name})
        else:
            out = self.transform(image=img, mask=mask, dataset=self)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_name})

        outputs.update({'indices': index})
        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
