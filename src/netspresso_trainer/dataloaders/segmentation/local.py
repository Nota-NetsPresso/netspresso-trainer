from functools import partial
import os
from pathlib import Path
from typing import Literal
from multiprocessing.pool import ThreadPool

import numpy as np
import PIL.Image as Image
from loguru import logger

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

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(Path(samples[i]['image'])).convert('RGB')
            label = self.samples[i]['label']
            if label is not None:
                label = Image.open(Path(label)).convert(self.label_image_mode)
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
            label = self.samples[index]['label']
        else:
            img_path = self.samples[index]['image']
            ann_path = self.samples[index]['label']
            img = Image.open(Path(img_path)).convert('RGB')
            label = Image.open(Path(ann_path)).convert(self.label_image_mode) if ann_path is not None else None

        w, h = img.size

        outputs = {}
        outputs.update({'indices': index})
        if label is None:
            out = self.transform(image=img)
            outputs.update({'pixel_values': out['image'], 'org_shape': (h, w)})
            return outputs

        
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
            out = self.transform(image=img, mask=mask, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float()})
        else:
            out = self.transform(image=img, mask=mask)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask']})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
