import os

import PIL.Image as Image

from dataloaders.base import BaseCustomDataset
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

class ClassificationCustomDataset(BaseCustomDataset):

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
        super(ClassificationCustomDataset, self).__init__(
            args,
            root,
            split,
            with_label
        )

        self.transform = transform

        self.samples = samples
        self.idx_to_class = idx_to_class
        self._num_classes = len(self.idx_to_class)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self.idx_to_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index]['image']
        target = self.samples[index]['label'] if 'label' in self.samples[index] else None
        img = Image.open(img).convert('RGB')
        
        if self.transform is not None:
            out = self.transform(args_augment=self.args.augment, img_size=self.args.training.img_size)(img)
        
        if target is None:
            target = -1  # To be ignored at cross-entropy loss
        return out['image'], target