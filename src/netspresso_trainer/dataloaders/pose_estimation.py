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
from functools import partial
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch.distributed as dist
from loguru import logger
from omegaconf import ListConfig

from .base import BaseCustomDataset, BaseSampleLoader
from .utils.constants import IMG_EXTENSIONS
from .utils.misc import natural_key


class PoseEstimationSampleLoader(BaseSampleLoader):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(PoseEstimationSampleLoader, self).__init__(conf_data, train_valid_split_ratio)

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

        else:
            for ext in IMG_EXTENSIONS:
                images_and_targets.extend([{'image': str(file), 'label': None}
                                        for file in chain(image_dir.glob(f'*{ext}'), image_dir.glob(f'*{ext.upper()}'))])
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k['image']))

        return images_and_targets

    def load_id_mapping(self):
        root_path = Path(self.conf_data.path.root)

        if isinstance(self.conf_data.id_mapping, ListConfig):
            return list(self.conf_data.id_mapping)
        elif isinstance(self.conf_data.id_mapping, str):
            id_mapping_path = root_path / self.conf_data.id_mapping
            with open(id_mapping_path, 'r') as f:
                id_mapping = json.load(f)
            return id_mapping
        else:
            raise ValueError(f"Invalid id_mapping type: {type(self.conf_data.id_mapping)}")

    def load_class_map(self, id_mapping):
        idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
        return {'idx_to_class': idx_to_class}

    def load_huggingface_samples(self):
        raise NotImplementedError


class PoseEstimationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, **kwargs):
        super(PoseEstimationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, **kwargs
        )
        flattened_samples = []
        # label field must be filled
        for sample in self.samples:
            flattened_sample = {}
            with open(sample['label'], 'r') as f:
                lines = f.readlines()
                f.close()
            flattened_sample = [{'image': sample['image'], 'label': line.strip()} for line in lines]
            flattened_samples += flattened_sample
        self.samples = flattened_samples

        # Build flip map. This is needed when try randomflip augmentation.
        if split == 'train':
            trasnform_names = {transform_conf['name'] for transform_conf in conf_augmentation[split]}
            flips = {'randomhorizontalflip', 'randomverticalflip'}
            if len(trasnform_names.intersection(flips)) > 0:
                class_to_idx = {self._idx_to_class[i]['name']: i for i in self._idx_to_class}
                self.flip_indices = np.zeros(self._num_classes).astype('int')
                for idx in self._idx_to_class:
                    idx_swap = self._idx_to_class[idx]['swap']
                    assert idx_swap is not None, "To apply flip transform, keypoint swap info must be filled."
                    self.flip_indices[idx] = class_to_idx[idx_swap] if idx_swap else -1

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(Path(samples[i]['image'])).convert('RGB')
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
        ann = self.samples[index]['label'] # TODO: Pose estimation is not assuming that label can be None now

        if not self.cache:
            img = Image.open(Path(img)).convert('RGB')

        w, h = img.size

        outputs = {}
        outputs.update({'indices': index})
        if ann is None:
            out = self.transform(image=img)
            outputs.update({'pixel_values': out['image'], 'org_shape': (h, w)})
            return outputs

        ann = ann.split(' ')
        bbox = ann[-4:]
        keypoints = ann[:-4]

        bbox = np.array(bbox).astype('float32')[np.newaxis, ...]
        keypoints = np.array(keypoints).reshape(-1, 3).astype('float32')[np.newaxis, ...]

        out = self.transform(image=img, bbox=bbox, keypoint=keypoints, dataset=self)

        # Use only one instance keypoints
        outputs.update({'pixel_values': out['image'], 'keypoints': out['keypoint'][0]})
        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
