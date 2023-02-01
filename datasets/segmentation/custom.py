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
            root,
            parser=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        super(SegmentationCustomDataset, self).__init__(
            root,
            parser,
            load_bytes,
            transform,
            target_transform
        )
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError