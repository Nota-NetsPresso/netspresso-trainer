import os

import PIL.Image as Image

from ..base import BaseCustomDataset


class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, with_label=True, **kwargs):
        super(ClassificationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, with_label, **kwargs
        )

    def __getitem__(self, index):
        img = self.samples[index]['image']
        target = self.samples[index]['label'] if 'label' in self.samples[index] else None
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            out = self.transform(conf_augmentation=self.conf_augmentation)(img)

        if target is None:
            target = -1  # To be ignored at cross-entropy loss
        return out['image'], target
