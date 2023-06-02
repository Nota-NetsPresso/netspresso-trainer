import os
from pathlib import Path
import logging
import json
from typing import List

import PIL.Image as Image
import numpy as np
from omegaconf import OmegaConf
import torch

from datasets.base import BaseCustomDataset

_logger = logging.getLogger(__name__)

ID2LABEL_FILENAME = "id2label.json"

TEMP_DIRECTORY_REDIRECT = lambda x: f"{x}2017"
TEMP_COCO_LABEL_FILE = "datasets/detection/coco.yaml"


def exist_name(candidate, folder_iterable):
    try:
        return list(filter(lambda x: candidate[0] in x, folder_iterable))[0]
    except:
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

    def __init__(
            self,
            args,
            root,
            split,
            transform=None,
            target_transform=None,
            load_bytes=False,
    ):
        super(DetectionCustomDataset, self).__init__(
            args,
            root,
            split
        )

        if self._split in ['train', 'training', 'val', 'valid', 'test']:  # for training and test (= evaluation) phase
            image_dir = Path(self._root) / 'images'
            annotation_dir = Path(self._root) / 'labels'
            self.image_dir = image_dir / TEMP_DIRECTORY_REDIRECT(self._split)
            self.annotation_dir = annotation_dir / TEMP_DIRECTORY_REDIRECT(self._split)

            img_name_maybe: List[Path] = list(sorted([path for path in self.image_dir.iterdir()]))
            # self.ann_name = list(sorted([path for path in self.annotation_dir.iterdir()]))
            self.img_name = []
            self.ann_name = []
            for image_path_maybe in img_name_maybe:
                ann_path_maybe = self.annotation_dir / image_path_maybe.with_suffix('.txt').name
                if not ann_path_maybe.exists():
                    continue
                self.img_name.append(image_path_maybe)
                self.ann_name.append(ann_path_maybe)
            
            del img_name_maybe
            self.id2label = OmegaConf.load(TEMP_COCO_LABEL_FILE)

            assert len(self.img_name) == len(self.ann_name), f"There must be as many images as there are detection label files! {len(self.img_name)}, {len(self.ann_name)}"

        else:  # self._split in ['infer', 'inference']

            try:  # a folder with multiple images
                self.img_name = list(sorted([path for path in Path(self._root).iterdir()]))
            except:  # single image
                raise AssertionError
                # TODO: check the case for single image
                self.file_name = [self.data_dir.split('/')[-1]]
                self.img_name = [self.data_dir]

        self.transform = transform
        self.load_bytes = load_bytes

    def __len__(self):
        return len(self.img_name)

    @property
    def num_classes(self):
        return self.id2label.num_classes
    
    @property
    def class_map(self):
        return {idx: name for idx, name in enumerate(self.id2label.names)}
    
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
        img_path = self.img_name[index]
        ann_path = self.ann_name[index]
        img = Image.open(str(img_path)).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if self._split in ['infer', 'inference']:
            out = self.transform(self.args.augment)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}
        
        outputs = {}

        label, boxes_yolo = get_label(Path(ann_path))
        boxes = self.xywhn2xyxy(boxes_yolo, w, h)
        
        out = self.transform(self.args.augment)(image=img, bbox=np.concatenate((boxes, label), axis=-1))
        assert out['bbox'].shape[-1] == 5  # ltrb + class_label
        outputs.update({'pixel_values': out['image'], 'bbox': out['bbox'][..., :4],
                        'label': torch.as_tensor(out['bbox'][..., 4], dtype=torch.int64)})


        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1 
        outputs.update({'org_shape': (h, w)})
        return outputs
    

def detection_collate_fn(original_batch):
    pixel_values = []
    bbox = []
    label = []
    org_shape = []
    
    for data_sample in original_batch:
        if 'pixel_values' in data_sample:
            pixel_values.append(data_sample['pixel_values'])
        if 'bbox' in data_sample:
            bbox.append(data_sample['bbox'])
        if 'label' in data_sample:
            label.append(data_sample['label'])
        if 'org_shape' in data_sample:
            org_shape.append(data_sample['org_shape'])
    
    outputs = {}
    if len(pixel_values) != 0:
        pixel_values = torch.stack(pixel_values, dim=0)
        outputs.update({'pixel_values': pixel_values})
    if len(bbox) != 0:
        bbox = torch.cat(bbox, dim=0)
        outputs.update({'bbox': bbox})
    if len(label) != 0:
        label = torch.cat(label, dim=0)
        outputs.update({'label': label})
    if len(org_shape) != 0:
        outputs.update({'org_shape': org_shape})

    return outputs
