import os
from pathlib import Path
import logging
import json

import cv2
import PIL.Image as Image
import numpy as np
import torch

from datasets.base import BaseCustomDataset
from datasets.segmentation.transforms import generate_edge

_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50

ID2LABEL_FILENAME = "id2label.json"


def exist_name(candidate, folder_iterable):
    try:
        return list(filter(lambda x: candidate[0] in x, folder_iterable))[0]
    except:
        return list(filter(lambda x: candidate[1] in x, folder_iterable))[0]


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


class SegmentationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            root,
            split,
            transform=None,
            target_transform=None,
            load_bytes=False,
    ):
        super(SegmentationCustomDataset, self).__init__(
            args,
            root,
            split
        )

        if self._split in ['train', 'training', 'val', 'valid', 'test']:  # for training and test (= evaluation) phase
            image_dir = Path(self._root) / 'image'
            annotation_dir = Path(self._root) / 'mask'
            if not annotation_dir.exists():
                annotation_dir = Path(self._root) / 'annotation'
            self.image_dir = image_dir / self._split
            self.annotation_dir = annotation_dir / self._split

            self.id2label = read_json(Path(self._root) / ID2LABEL_FILENAME)

            self.img_name = list(sorted([path for path in self.image_dir.iterdir()]))
            self.ann_name = list(sorted([path for path in self.annotation_dir.iterdir()]))

            assert len(self.img_name) == len(self.ann_name), "There must be as many images as there are segmentation maps"

        else:  # self._split in ['infer', 'inference']

            try:  # a folder with multiple images
                self.img_name = list(sorted([path for path in Path(self.data_dir).iterdir()]))
            except:  # single image
                raise AssertionError
                # TODO: check the case for single image
                self.file_name = [self.data_dir.split('/')[-1]]
                self.img_name = [self.data_dir]

        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.load_bytes = load_bytes

    def __len__(self):
        return len(self.img_name)

    @ property
    def num_classes(self):
        return len(self.id2label)

    def __getitem__(self, index):
        img_path = self.img_name[index]
        ann_path = self.ann_name[index]
        img = np.array(Image.open(str(img_path)).convert('RGB'))

        org_img = img.copy()

        h, w = img.shape[:2]

        if self._split in ['infer', 'inference']:
            out = self.transform(self.args.augment, (h, w), label, use_prefetcher=True)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}
        
        outputs = {}

        label = np.array(Image.open(str(ann_path)).convert('L'))
        if self.args.train.architecture.full == 'pidnet':
            edge = generate_edge(label)
            out = self.transform(self.args.augment, (h, w), label, use_prefetcher=True)(image=img, mask=label, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'], 'name': img_path.name})
        else:
            out = self.transform(self.args.augment, (h, w), label, use_prefetcher=True)(image=img, mask=label)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        outputs.update({'org_img': org_img, 'org_shape': (h, w)})
        return outputs
