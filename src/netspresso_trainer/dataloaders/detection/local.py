import os
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image as Image
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf

from ..base import BaseCustomDataset

ID2LABEL_FILENAME = "id2label.json"
TEMP_COCO_LABEL_FILE = "data/detection/coco.yaml"


def exist_name(candidate, folder_iterable):
    try:
        return list(filter(lambda x: candidate[0] in x, folder_iterable))[0]
    except IndexError:
        return list(filter(lambda x: candidate[1] in x, folder_iterable))[0]


def get_label(label_file: Path):
    target = Path(label_file).read_text()

    if target == '': # target label can be empty string
        target_array = np.zeros((0, 5))
    else:
        try:
            target_array = np.array([list(map(float, box.split(' '))) for box in target.split('\n') if box.strip()])
        except ValueError as e:
            print(target)
            raise e

    label, boxes = target_array[:, 0], target_array[:, 1:]
    label = label[..., np.newaxis]
    return label, boxes


class DetectionCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, **kwargs):
        super(DetectionCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, **kwargs,
        )

    @staticmethod
    def xywhn2xyxy(original: np.ndarray, w: int, h: int, padw=0, padh=0):
        converted = original.copy()
        # left, top (lt)
        converted[..., 0] = w * (original[..., 0] - original[..., 2] / 2) + padw
        converted[..., 1] = h * (original[..., 1] - original[..., 3] / 2) + padh
        # right, bottom (rb)
        converted[..., 2] = w * (original[..., 0] + original[..., 2] / 2) + padw
        converted[..., 3] = h * (original[..., 1] + original[..., 3] / 2) + padh
        return converted

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(Path(samples[i]['image'])).convert('RGB')
            label = self.samples[i]['label']
            if label is not None:
                label = get_label(Path(label))
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
            ann = self.samples[index]['label']
        else:
            img = Image.open(self.samples[index]['image']).convert('RGB')
            ann_path = Path(self.samples[index]['label']) if self.samples[index]['label'] is not None else None
            ann = get_label(Path(ann_path)) if ann_path is not None else None

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

    def pull_item(self, index):
        img_path = Path(self.samples[index]['image'])
        ann_path = Path(self.samples[index]['label']) if 'label' in self.samples[index] else None
        img = Image.open(str(img_path)).convert('RGB')

        org_img = img.copy()
        w, h = img.size
        if ann_path is None:
            return org_img, np.zeros(0, 1), np.zeros(0, 5)

        label, boxes_yolo = get_label(Path(ann_path))
        boxes = self.xywhn2xyxy(boxes_yolo, w, h)

        return org_img, label, boxes

