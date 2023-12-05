from typing import Literal

import numpy as np
import PIL.Image as Image

from ..augmentation.transforms import generate_edge, reduce_label
from ..base import BaseHFDataset


class SegmentationHFDataset(BaseHFDataset):

    def __init__(
            self,
            conf_data,
            conf_augmentation,
            model_name,
            idx_to_class,
            split,
            huggingface_dataset,
            transform=None,
            with_label=True,
            **kwargs
    ):
        root = conf_data.metadata.repo
        super(SegmentationHFDataset, self).__init__(
            conf_data,
            conf_augmentation,
            model_name,
            root,
            split,
            with_label
        )

        self.transform = transform
        self.idx_to_class = idx_to_class
        self.samples = huggingface_dataset

        assert "label_value_to_idx" in kwargs
        self.label_value_to_idx = kwargs["label_value_to_idx"]

        self.label_image_mode: Literal['RGB', 'L', 'P'] = str(conf_data.label_image_mode).upper() \
            if conf_data.label_image_mode is not None else 'L'

        self.image_feature_name = conf_data.metadata.features.image
        self.label_feature_name = conf_data.metadata.features.label

    @property
    def num_classes(self):
        return len(self.idx_to_class)

    @property
    def class_map(self):
        return self.idx_to_class

    def __len__(self):
        return self.samples.num_rows

    def __getitem__(self, index):

        img_name = f"{index:06d}"
        img: Image.Image = self.samples[index][self.image_feature_name]
        label: Image.Image = self.samples[index][self.label_feature_name] if self.label_feature_name in self.samples[index] else None

        label_array = np.array(label.convert(self.label_image_mode))
        label_array = label_array[..., np.newaxis] if label_array.ndim == 2 else label_array
        # if self.conf_augmentation.reduce_zero_label:
        #     label = reduce_label(np.array(label))

        mask = np.zeros((label.size[1], label.size[0]), dtype=np.uint8)
        for label_value in self.label_value_to_idx:
            class_mask = (label_array == np.array(label_value)).all(axis=-1)
            mask[class_mask] = self.label_value_to_idx[label_value]
        mask = Image.fromarray(mask, mode='L')  # single mode array (PIL.Image) compatbile with torchvision transform API

        org_img = img.copy()
        w, h = img.size

        if label is None:
            out = self.transform(self.conf_augmentation)(image=img)
            return {'pixel_values': out['image'], 'name': img_name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        if self.model_name == 'pidnet':
            edge = generate_edge(np.array(label))
            out = self.transform(self.conf_augmentation)(image=img, mask=mask, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_name})
        else:
            out = self.transform(self.conf_augmentation)(image=img, mask=mask)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
