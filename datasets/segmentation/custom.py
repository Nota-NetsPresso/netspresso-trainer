import os
from pathlib import Path
import logging

import PIL.Image as Image
import torch
import torch.utils.data as data

from datasets.base import BaseCustomDataset

_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50

class SegmentationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            root,
            split,
            transform=None,
            target_transform=None,
            load_bytes=False,
    ):
        super(SegmentationCustomDataset, self).__init__(
            args,
            root,
            split
        )

        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.load_bytes = load_bytes

    def __getitem__(self, index):
        raise NotImplementedError
