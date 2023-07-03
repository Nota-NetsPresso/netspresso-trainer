import os
from typing import Union

import PIL.Image as Image

from dataloaders.base import BaseHFDataset
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

class ClassificationHFDataset(BaseHFDataset):

    def __init__(
            self,
            args,
            idx_to_class,
            split,
            huggingface_dataset,
            transform=None,
    ):
        root = args.data.metadata.repo
        super(ClassificationHFDataset, self).__init__(
            args,
            root,
            split
        )
        # Make sure that you additionally install `requirements-data.txt`

        self.transform = transform
        
        self.samples = huggingface_dataset
        self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        
        self.image_feature_name = args.data.metadata.features.image
        self.label_feature_name = args.data.metadata.features.label
        
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
            out = self.transform(args_augment=self.args.augment, img_size=self.args.training.img_size)(img)
        if target is None:
            target = -1
        return out['image'], target