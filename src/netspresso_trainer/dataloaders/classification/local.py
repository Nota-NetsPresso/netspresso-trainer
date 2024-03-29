import os

import PIL.Image as Image
from loguru import logger

from ..base import BaseCustomDataset
import torch.distributed as dist


class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, with_label=True, **kwargs):
        super(ClassificationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, with_label, **kwargs
        )

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')
        for i in sampler:
            self.samples[i]['image'] = Image.open(str(self.samples[i]['image'])).convert('RGB')
            self.cache = True

    def __getitem__(self, index):
        img = self.samples[index]['image']
        target = self.samples[index]['label']
        if not self.cache:
            img = Image.open(img).convert('RGB')

        if self.transform is not None:
            out = self.transform(img)

        if target is None:
            target = -1  # To be ignored at cross-entropy loss
        return index, out['image'], target
