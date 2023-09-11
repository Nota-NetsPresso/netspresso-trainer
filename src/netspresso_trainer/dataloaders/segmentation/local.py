import os
from pathlib import Path

import numpy as np
import PIL.Image as Image

from ..base import BaseCustomDataset
from ..segmentation.transforms import generate_edge, reduce_label


class SegmentationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, with_label=True):
        super(SegmentationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, with_label
        )

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

        label = Image.open(ann_path).convert('L')
        # if self.conf_augmentation.reduce_zero_label:
        #     label = reduce_label(np.array(label))

        if self.model_name == 'pidnet':
            edge = generate_edge(np.array(label))
            out = self.transform(self.conf_augmentation)(image=img, mask=label, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_path.name})
        else:
            out = self.transform(self.conf_augmentation)(image=img, mask=label)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs