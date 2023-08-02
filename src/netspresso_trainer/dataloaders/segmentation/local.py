import os
from pathlib import Path

import PIL.Image as Image
import numpy as np

from dataloaders.base import BaseCustomDataset
from dataloaders.segmentation.transforms import generate_edge, reduce_label
from utils.logger import set_logger

logger = set_logger('data', level=os.getenv('LOG_LEVEL', default='INFO'))


class SegmentationCustomDataset(BaseCustomDataset):

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
        super(SegmentationCustomDataset, self).__init__(
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
        img_path = Path(self.samples[index]['image'])
        ann_path = Path(self.samples[index]['label']) if 'label' in self.samples[index] else None
        img = Image.open(img_path).convert('RGB')

        org_img = img.copy()

        w, h = img.size

        if ann_path is None:
            out = self.transform(self.args.augment)(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        label = Image.open(ann_path).convert('L')
        # if self.args.augment.reduce_zero_label:
        #     label = reduce_label(np.array(label))

        if self.args.model.architecture.full == 'pidnet':
            edge = generate_edge(np.array(label))
            out = self.transform(self.args.augment)(image=img, mask=label, edge=edge)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'edges': out['edge'].float(), 'name': img_path.name})
        else:
            out = self.transform(self.args.augment)(image=img, mask=label)
            outputs.update({'pixel_values': out['image'], 'labels': out['mask'], 'name': img_path.name})

        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs