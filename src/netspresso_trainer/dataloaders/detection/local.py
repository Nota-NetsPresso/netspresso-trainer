import os
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image as Image
import torch
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
                 split, samples, transform=None, with_label=True, **kwargs):
        super(DetectionCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, with_label, **kwargs
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

    def __getitem__(self, index):
        img_path = Path(self.samples[index]['image'])
        ann_path = Path(self.samples[index]['label']) if self.samples[index]['label'] is not None else None
        img = Image.open(str(img_path)).convert('RGB')

        w, h = img.size

        outputs = {}
        outputs.update({'indices': index})
        if ann_path is None:
            out = self.transform(image=img)
            outputs.update({'pixel_values': out['image'], 'name': img_path.name, 'org_shape': (h, w)})
            return outputs

        label, boxes_yolo = get_label(Path(ann_path))
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
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
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

