import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import PIL.Image as Image

from dataloaders.utils.parsers import create_parser
from dataloaders.base import BaseCustomDataset, BaseHFDataset
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))

_ERROR_RETRY = 50

class ClassificationCustomDataset(BaseCustomDataset):

    def __init__(
            self,
            args,
            split,
            transform=None,
            target_transform=None,
    ):
        root = args.data.path.root
        super(ClassificationCustomDataset, self).__init__(
            args,
            root,
            split
        )

        self.transform = transform
        self._consecutive_errors = 0

        _class_map: Optional[dict] = dict(self.args.data.id_mapping) \
            if self.args.data.id_mapping is not None else None

        self.parser = create_parser(name='', root=self._root, split=self._split, class_map=_class_map)
        self._num_classes = self.parser.num_classes

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
            img = Image.open(img).convert('RGB')
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


class ClassificationHFDataset(BaseHFDataset):

    def __init__(
            self,
            args,
            split,
            transform=None,
            target_transform=None,
    ):
        root = args.data.metadata.repo
        super(ClassificationHFDataset, self).__init__(
            args,
            root,
            split
        )
        # Make sure that you additionally install `requirements-data.txt`

        self.transform = transform
        
        subset_name = args.data.metadata.subset
        cache_dir = args.data.metadata.custom_cache_dir
        total_dataset = self._load_dataset(root, subset_name=subset_name, cache_dir=cache_dir)

        if self._split in ['train', 'training']:
            _dataset = total_dataset['train']
        elif self._split in ['val', 'valid', 'validation']:
            _dataset = total_dataset['validation']
        else:
            raise NotImplementedError
            
        image_key = args.data.metadata.features.image
        label_key = args.data.metadata.features.label
            
        self._class_map: Dict[int, str] = {class_idx: str(class_name)
                                      for class_idx, class_name
                                      in enumerate(sorted(set([x[label_key] for x in _dataset])))}
        self._class_name_as_index: Dict[str, int] = {str(v): k for k, v in self._class_map.items()}
            
        self._num_classes = len(self._class_map)
        
        self.parser: Dict[Tuple[Image.Image, int]] = [(x[image_key], self._class_name_as_index[str(x[label_key])]) for x in _dataset]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_map(self):
        return self._class_map

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, index):
        img, target = self.parser[index]
        if self.transform is not None:
            out = self.transform(args_augment=self.args.augment, img_size=self.args.training.img_size)(img)
        if target is None:
            target = -1
        return out['image'], target

def create_classification_dataset(args, split, transform, target_transform=None):
    data_format = args.data.format
    if data_format == 'local':
        return ClassificationCustomDataset(args, split=split,
                                           transform=transform, target_transform=target_transform)
    elif data_format == 'huggingface':
        return ClassificationHFDataset(args, split=split,
                                       transform=transform, target_transform=target_transform)
    else:
        raise AssertionError(f"No such data format named {data_format}!")