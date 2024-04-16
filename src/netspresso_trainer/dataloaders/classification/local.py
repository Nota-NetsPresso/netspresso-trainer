import os
from functools import partial
from multiprocessing.pool import ThreadPool

import PIL.Image as Image
import torch.distributed as dist
from loguru import logger

from ..base import BaseCustomDataset


class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, **kwargs):
        super(ClassificationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, **kwargs
        )

    def cache_dataset(self, sampler, distributed):
        if (not distributed) or (distributed and dist.get_rank() == 0):
            logger.info(f'Caching | Loading samples of {self.mode} to memory... This can take minutes.')

        def _load(i, samples):
            image = Image.open(str(samples[i]['image'])).convert('RGB')
            return i, image

        num_threads = 8 # TODO: Compute appropriate num_threads
        load_imgs = ThreadPool(num_threads).imap(
            partial(_load, samples=self.samples),
            sampler
        )
        for i, image in load_imgs:
            self.samples[i]['image'] = image

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
