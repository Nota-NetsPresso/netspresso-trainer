import os
from pathlib import Path
from typing import List

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
        return index
