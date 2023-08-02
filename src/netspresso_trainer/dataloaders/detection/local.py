import os
from pathlib import Path
import logging
from typing import List

import PIL.Image as Image
import numpy as np
from omegaconf import OmegaConf
import torch

from ..base import BaseCustomDataset
from ...utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

ID2LABEL_FILENAME = "id2label.json"

TEMP_DIRECTORY_REDIRECT = lambda x: f"{x}2017"
TEMP_COCO_LABEL_FILE = "data/detection/coco.yaml"


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
            idx_to_class,
            split,
            samples,
            transform=None,
            with_label=True,
    ):
        root = args.data.path.root
        super(DetectionCustomDataset, self).__init__(
            args,
            root,
            split,
            with_label
        )
        
        self.transform = transform

        self.samples = samples
        self.idx_to_class = idx_to_class
        self._num_classes = len(self.idx_to_class)

        # if self._split in ['train', 'training', 'val', 'valid', 'test']:  # for training and test (= evaluation) phase
        #     image_dir = Path(self._root) / 'images'
        #     annotation_dir = Path(self._root) / 'labels'
        #     self.image_dir = image_dir / TEMP_DIRECTORY_REDIRECT(self._split)
        #     self.annotation_dir = annotation_dir / TEMP_DIRECTORY_REDIRECT(self._split)

        #     img_name_maybe: List[Path] = list(sorted([path for path in self.image_dir.iterdir()]))
        #     # self.ann_name = list(sorted([path for path in self.annotation_dir.iterdir()]))
        #     self.img_name = []
        #     self.ann_name = []
        #     for image_path_maybe in img_name_maybe:
        #         ann_path_maybe = self.annotation_dir / image_path_maybe.with_suffix('.txt').name
        #         if not ann_path_maybe.exists():
        #             continue
        #         self.img_name.append(image_path_maybe)
        #         self.ann_name.append(ann_path_maybe)
            
        #     del img_name_maybe
        #     self.id2label = OmegaConf.load(TEMP_COCO_LABEL_FILE)

        #     assert len(self.img_name) == len(self.ann_name), f"There must be as many images as there are detection label files! {len(self.img_name)}, {len(self.ann_name)}"

        # else:  # self._split in ['infer', 'inference']

        #     try:  # a folder with multiple images
        #         self.img_name = list(sorted([path for path in Path(self._root).iterdir()]))
        #     except:  # single image
        #         raise AssertionError
        #         # TODO: check the case for single image
        #         self.file_name = [self.data_dir.split('/')[-1]]
        #         self.img_name = [self.data_dir]

        # self.transform = transform

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self.idx_to_class
    
    def __len__(self):
        return len(self.samples)
    
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
        ann_path = Path(self.samples[index]['label']) if 'label' in self.samples[index] else None
        img = Image.open(str(img_path)).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if ann_path is None:
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