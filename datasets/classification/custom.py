import os
from pathlib import Path
import logging

import PIL.Image as Image
import torch
import torch.utils.data as data

from datasets.utils.parsers import create_parser
from datasets.base import BaseCustomDataset

_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50

class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            root,
            parser=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        super(ClassificationCustomDataset, self).__init__(
            args,
            root,
            parser,
            load_bytes,
            transform,
            target_transform
        )
        
        self.class_map = args.class_map if Path(args.class_map).exists() else None

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target