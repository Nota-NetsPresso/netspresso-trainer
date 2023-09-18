import os
from typing import Union

import PIL.Image as Image

from ..base import BaseHFDataset


class ClassificationHFDataset(BaseHFDataset):

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
        super(ClassificationHFDataset, self).__init__(
            conf_data,
            conf_augmentation,
            model_name,
            root,
            split,
            with_label
        )
        # Make sure that you additionally install `requirements-data.txt`

        self.transform = transform

        self.samples = huggingface_dataset
        self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}

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
        img: Image.Image = self.samples[index][self.image_feature_name]
        target: Union[int, str] = self.samples[index][self.label_feature_name] if self.label_feature_name in self.samples[index] else None
        if isinstance(target, str):
            target: int = self.class_to_idx[target]

        if self.transform is not None:
            out = self.transform(conf_augmentation=self.conf_augmentation)(img)
        if target is None:
            target = -1
        return out['image'], target
