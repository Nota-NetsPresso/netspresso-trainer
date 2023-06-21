import os
from pathlib import Path
import logging
import json

import PIL.Image as Image
import numpy as np

from dataloaders.base import BaseCustomDataset
from dataloaders.segmentation.transforms import generate_edge, reduce_label

_logger = logging.getLogger(__name__)


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
            self.image_dir = Path(self._root) / args.data.path.train.image
            self.annotation_dir = Path(self._root) / args.data.path.train.mask

            self.id2label = args.data.id_mapping

            self.img_name = list(sorted([path for path in self.image_dir.iterdir()]))
            self.ann_name = list(sorted([path for path in self.annotation_dir.iterdir()]))
            # TODO: get paired data from regex pattern matching (args.data.path.pattern)

            assert len(self.img_name) == len(self.ann_name), "There must be as many images as there are segmentation maps"

        else:  # self._split in ['infer', 'inference']
            raise NotImplementedError
            try:  # a folder with multiple images
                self.img_name = list(sorted([path for path in Path(self.data_dir).iterdir()]))
            except:  # single image
                raise AssertionError
                # TODO: check the case for single image
                self.file_name = [self.data_dir.split('/')[-1]]
                self.img_name = [self.data_dir]

        self.transform = transform
        self._consecutive_errors = 0
        self.load_bytes = load_bytes

    def __len__(self):
        return len(self.img_name)

    @property
    def num_classes(self):
        return len(self.id2label)

    @property
    def class_map(self):
        return self.id2label

    def __getitem__(self, index):
        img_path = self.img_name[index]
        ann_path = self.ann_name[index]
        img = Image.open(str(img_path)).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if self._split in ['infer', 'inference']:
            out = self.transform(self.args.augment)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        label = Image.open(str(ann_path)).convert('L')
        if self.args.augment.reduce_zero_label:
            label = reduce_label(np.array(label))

        if self.args.train.architecture.full == 'pidnet':
            edge = generate_edge(np.array(label))
            out = self.transform(self.args.augment)(image=img, mask=label, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_path.name})
        else:
            out = self.transform(self.args.augment)(image=img, mask=label)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
