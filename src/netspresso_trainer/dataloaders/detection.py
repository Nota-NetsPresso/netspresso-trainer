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
from collections import Counter
from functools import partial
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import ListConfig

from .base import BaseCustomDataset, BaseSampleLoader
from .utils.constants import IMG_EXTENSIONS
from .utils.misc import get_detection_label, natural_key


class DetectionSampleLoader(BaseSampleLoader):
    def __init__(self, conf_data, train_valid_split_ratio):
        super(DetectionSampleLoader, self).__init__(conf_data, train_valid_split_ratio)

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
            if not os.path.isfile(id_mapping_path):
                raise FileNotFoundError(f"File not found: {id_mapping_path}")

            with open(id_mapping_path, 'r') as f:
                id_mapping = json.load(f)
            return id_mapping

        else:
            raise ValueError(f"Unsupported id_mapping value {self.conf_data.id_mapping}")

    def load_class_map(self, id_mapping):
        idx_to_class: Dict[int, str] = dict(enumerate(id_mapping))
        return {'idx_to_class': idx_to_class}

    def load_huggingface_samples(self):
        raise NotImplementedError


class DetectionCustomDataset(BaseCustomDataset):
    __repr_indent = 4

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, **kwargs):
        super(DetectionCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, **kwargs,
        )
        self._stats = self.get_stats()

    @staticmethod
    def xywhn2xyxy(original: np.ndarray, w: int, h: int, padw=0, padh=0):
        converted = original.copy()
        # left, top (lt)
        converted[..., 0] = w * (original[..., 0] - original[..., 2] / 2) + padw
        converted[..., 1] = h * (original[..., 1] - original[..., 3] / 2) + padh
        # right, bottom (rb)
        converted[..., 2] = w * (original[..., 0] + original[..., 2] / 2) + padw
        converted[..., 3] = h * (original[..., 1] + original[..., 3] / 2) + padh

        # bbox clamping
        np.clip(converted[..., 0::2], a_min=0, a_max=w, out=converted[..., 0::2])
        np.clip(converted[..., 1::2], a_min=0, a_max=h, out=converted[..., 1::2])
        return converted

    def cache_dataset(self):
        logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(Path(samples[i]['image'])).convert('RGB')
            label = self.samples[i]['label']
            if label is not None:
                label = get_detection_label(Path(label))
            return i, image, label

        all_indices = list(range(len(self.samples)))

        num_threads = 8 # TODO: Compute appropriate num_threads
        load_imgs = ThreadPool(num_threads).imap(
            partial(_load, samples=self.samples),
            all_indices
        )
        for i, image, label in load_imgs:
            self.samples[i]['image'] = image
            self.samples[i]['label'] = label

        self.cache = True

    def __getitem__(self, index):
        if self.cache:
            img = self.samples[index]['image']
            ann = self.samples[index]['label']
        else:
            img = Image.open(self.samples[index]['image']).convert('RGB')
            ann_path = Path(self.samples[index]['label']) if self.samples[index]['label'] is not None else None
            ann = get_detection_label(Path(ann_path)) if ann_path is not None else None

        w, h = img.size

        outputs = {}
        outputs.update({'indices': index})
        if ann is None:
            out = self.transform(image=img)
            outputs.update({'pixel_values': out['image'], 'org_shape': (h, w)})
            return outputs

        label, boxes_yolo = ann
        boxes = self.xywhn2xyxy(boxes_yolo, w, h)

        out = self.transform(image=img, label=label, bbox=boxes, dataset=self)
        # Remove
        mask = np.minimum(out['bbox'][:, 2] - out['bbox'][:, 0], out['bbox'][:, 3] - out['bbox'][:, 1]) > 1
        out['bbox'] = out['bbox'][mask]
        out['label'] = torch.as_tensor(out['label'].ravel(), dtype=torch.int64)
        out['label'] = out['label'][mask]
        outputs.update({'pixel_values': out['image'], 'bbox': out['bbox'],
                        'label': out['label']})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        outputs.update({'org_shape': (h, w)})
        return outputs

    def __repr__(self) -> str:
        '''
        This implementation is based on https://pytorch.org/vision/main/_modules/torchvision/datasets/vision.html#VisionDataset.
        '''
        head = "Dataset " + self.conf_data.name
        body = [] if self.root is None else [f"Root location: {self.root}"]
        body.append(f"Number of images: {self.__len__()}")
        if self.num_classes is not None:
            body.append(f"Number of classes: {self.num_classes}")
        if self.stats:
            body.append(f"Total instances: {self.stats['total_instances']}")
            body.append("Instances per class:")
            for class_idx, count in self.stats['instances_per_class'].items():
                percentage = (count / self.stats['total_instances']) * 100
                body.append(" " * self.__repr_indent + f"{class_idx}_{self._idx_to_class[class_idx]}: {count} ({percentage:.1f}%)")

        lines = [head] + [" " * self.__repr_indent + line for line in body]
        return "\n".join(lines)


    def pull_item(self, index):
        sample = self.samples[index]
        has_label = 'label' in sample

        if self.cache:
            img = sample['image']
            ann = sample['label'] if has_label else None
        else:
            img_path = Path(sample['image'])
            img = Image.open(str(img_path)).convert('RGB')
            if has_label:
                ann_path = Path(sample['label'])
                ann = get_detection_label(ann_path)
            else:
                ann = None

        org_img = img.copy()
        w, h = img.size

        if ann is None:
            return org_img, np.zeros((0, 1)), np.zeros((0, 5))

        label, boxes_yolo = ann
        boxes = self.xywhn2xyxy(boxes_yolo, w, h)

        return org_img, label, boxes

    def get_stats(self):
        def process_annotation(sample):
            try:
                ann_path = sample.get('label')
                if not ann_path:
                    return []

                label, _ = get_detection_label(Path(ann_path))
                label = label.squeeze()
                if label.ndim == 0:
                    return [int(label)]
                return label.astype(int).tolist()
            except Exception as e:
                logger.warning(f"Error processing annotation {ann_path}: {str(e)}")
                return []
        num_threads = min(8, len(self.samples))
        with ThreadPool(num_threads) as pool:
            all_labels = pool.map(process_annotation, self.samples)
        flat_labels = [label for sublist in all_labels for label in sublist]
        class_counts = dict(Counter(flat_labels))
        if not class_counts:
            return
        for i in range(len(self._idx_to_class)):
            if i not in class_counts:
                class_counts[i] = 0
        sorted_class_counts = dict(sorted(class_counts.items()))
        stats = {
            'class_distribution': class_counts,
            'total_instances': sum(class_counts.values()),
            'instances_per_class': dict(sorted_class_counts.items())
        }

        return stats
