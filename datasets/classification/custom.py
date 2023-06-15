import os
from pathlib import Path
from typing import Optional

import PIL.Image as Image

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
        self._consecutive_errors = 0

        _class_map: Optional[dict] = dict(self.args.datasets.id_mapping) \
            if self.args.datasets.id_mapping is not None else None

        self.parser = create_parser(name='', root=self._root, split=self._split, class_map=_class_map)
        self._num_classes = self.parser.num_classes

        self.load_bytes = load_bytes

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self.parser.idx_to_class

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
            out = self.transform(args_augment=self.args.augment, img_size=self.args.training.img_size)(img)
        if target is None:
            target = -1
        return out['image'], target
