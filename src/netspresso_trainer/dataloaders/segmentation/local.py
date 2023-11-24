import os
from pathlib import Path
from typing import Literal

import numpy as np
import PIL.Image as Image

from ..augmentation.transforms import generate_edge, reduce_label
from ..base import BaseCustomDataset


class SegmentationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, with_label=True, **kwargs):
        super(SegmentationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, with_label, **kwargs
        )
        assert "label_value_to_idx" in kwargs
        self.label_value_to_idx = kwargs["label_value_to_idx"]

        self.label_image_mode: Literal['RGB', 'L', 'P'] = str(conf_data.label_image_mode).upper() \
            if conf_data.label_image_mode is not None else 'L'

    def __getitem__(self, index):
        img_path = Path(self.samples[index]['image'])
        ann_path = Path(self.samples[index]['label']) if 'label' in self.samples[index] else None
        img = Image.open(img_path).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if ann_path is None:
            out = self.transform(self.conf_augmentation)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        label = Image.open(ann_path).convert(self.label_image_mode)
        label_array = np.array(label)
        label_array = label_array[..., np.newaxis] if label_array.ndim == 2 else label_array
        # if self.conf_augmentation.reduce_zero_label:
        #     label = reduce_label(np.array(label))

        mask = np.zeros((label.size[1], label.size[0]), dtype=np.uint8)
        for label_value in self.label_value_to_idx:
            class_mask = (label_array == np.array(label_value)).all(axis=-1)
            mask[class_mask] = self.label_value_to_idx[label_value]

        mask = Image.fromarray(mask, mode='L')  # single mode array (PIL.Image) compatbile with torchvision transform API

        if 'pidnet' in self.model_name:
            edge = generate_edge(np.array(mask))
            out = self.transform(self.conf_augmentation)(image=img, mask=mask, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_path.name})
        else:
            out = self.transform(self.conf_augmentation)(image=img, mask=mask)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
