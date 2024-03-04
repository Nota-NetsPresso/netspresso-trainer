import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import PIL.Image as Image
import torch
from omegaconf import OmegaConf

from ..base import BaseCustomDataset


class PoseEstimationCustomDataset(BaseCustomDataset):

    def __init__(self, conf_data, conf_augmentation, model_name, idx_to_class,
                 split, samples, transform=None, with_label=True, **kwargs):
        super(PoseEstimationCustomDataset, self).__init__(
            conf_data, conf_augmentation, model_name, idx_to_class,
            split, samples, transform, with_label, **kwargs
        )
        flattened_samples = []
        for sample in self.samples:
            flattened_sample = {}
            with open(sample['label'], 'r') as f:
                lines = f.readlines()
                f.close()
            flattened_sample = [{'image': sample['image'], 'label': line.strip()} for line in lines]
            flattened_samples += flattened_sample
        self.samples = flattened_samples

    def __getitem__(self, index):
        img_path = Path(self.samples[index]['image'])
        ann = self.samples[index]['label'] if 'label' in self.samples[index] else None

        img = Image.open(img_path).convert('RGB')

        org_img = img.copy()
        w, h = img.size

        if ann is None:
            out = self.transform(image=img)
            return {'pixel_values': out['image'], 'name': img_path.name, 'org_img': org_img, 'org_shape': (h, w)}

        outputs = {}

        ann = ann.split(' ')
        bbox = ann[-4:]
        keypoints = ann[:-4]

        bbox = np.array(bbox).astype('float32')
        keypoints = np.array(keypoints).reshape(-1, 3).astype('float32')

        # TODO: Apply transforms
        #out = self.transform(image=img, label=label, bbox=boxes, dataset=self)
        # Altenatively, just crop and resize. This must be fixed to apply transforms
        img = np.array(img)
        img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        crop_h, crop_w, _ = img.shape
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = torch.tensor(img).to(torch.float32) / 255.0
        img = torch.permute(img, (2, 0, 1))

        h_ratio = 256 / crop_h
        w_ratio = 256 / crop_w
        keypoints[:, 0] -= bbox[0]
        keypoints[:, 0] *= w_ratio

        keypoints[:, 1] -= bbox[1]
        keypoints[:, 1] *= h_ratio

        outputs.update({'pixel_values': img, 'keypoints': keypoints})

        outputs.update({'indices': index})
        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
