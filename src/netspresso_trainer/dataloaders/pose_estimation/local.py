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

        # Build flip map. This is needed when try randomflip augmentation.
        if split == 'train':
            trasnform_names = {transform_conf['name'] for transform_conf in conf_augmentation[split]}
            flips = {'randomhorizontalflip', 'randomverticalflip'}
            if len(trasnform_names.intersection(flips)) > 0:
                class_to_idx = {self._idx_to_class[i]['name']: i for i in self._idx_to_class}
                self.flip_indices = np.zeros(self._num_classes).astype('int')
                for idx in self._idx_to_class:
                    idx_swap = self._idx_to_class[idx]['swap']
                    assert idx_swap is not None, "To apply flip transform, keypoint swap info must be filled."
                    self.flip_indices[idx] = class_to_idx[idx_swap] if idx_swap else -1

    def __getitem__(self, index):
        img_path = Path(self.samples[index]['image'])
        ann = self.samples[index]['label'] # TODO: Pose estimation is not assuming that label can be None now

        img = Image.open(img_path).convert('RGB')

        w, h = img.size

        outputs = {}
        outputs.update({'indices': index})
        if ann is None:
            out = self.transform(image=img)
            outputs.update({'pixel_values': out['image'], 'name': img_path.name, 'org_shape': (h, w)})
            return outputs

        ann = ann.split(' ')
        bbox = ann[-4:]
        keypoints = ann[:-4]

        bbox = np.array(bbox).astype('float32')[np.newaxis, ...]
        keypoints = np.array(keypoints).reshape(-1, 3).astype('float32')[np.newaxis, ...]

        out = self.transform(image=img, bbox=bbox, keypoint=keypoints, dataset=self)

        # Use only one instance keypoints
        outputs.update({'pixel_values': out['image'], 'keypoints': out['keypoint'][0]})
        if self._split in ['train', 'training']:
            return outputs

        assert self._split in ['val', 'valid', 'test']
        # outputs.update({'org_img': org_img, 'org_shape': (h, w)})  # TODO: return org_img with batch_size > 1
        outputs.update({'org_shape': (h, w)})
        return outputs
