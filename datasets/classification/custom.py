import os
from pathlib import Path
import logging

import PIL.Image as Image
import torch

from datasets.utils.parsers import create_parser
from datasets.base import BaseCustomDataset
from utils.logger import set_logger

logger = set_logger('datasets', level=os.getenv('LOG_LEVEL', default='INFO'))

_ERROR_RETRY = 50
_MAPPING_TXT_FILE = "mapping.txt"


class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            root,
            split,
            transform=None,
            target_transform=None,
            load_bytes=False,
    ):
        super(ClassificationCustomDataset, self).__init__(
            args,
            root,
            split
        )

        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

        _class_map_maybe = Path(self.args.train.data) / _MAPPING_TXT_FILE
        class_map = _class_map_maybe if _class_map_maybe.exists() else None

        self.parser = create_parser(name='', root=self._root, split=self._split, class_map=class_map)
        self._num_classes = self.parser.num_classes

        self.load_bytes = load_bytes

    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def class_map(self):
        return self.parser.get_idx_to_class()

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            logger.warning(f"Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}")
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(use_prefetcher=True, img_size=self.args.train.img_size)(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(use_prefetcher=True, img_size=self.args.train.img_size)(target)
        return img, target
