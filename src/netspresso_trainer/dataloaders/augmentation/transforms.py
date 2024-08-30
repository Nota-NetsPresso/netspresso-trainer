# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from functools import partial

import cv2
import numpy as np
import PIL.Image as Image

from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .custom import image_proc as TC
from .registry import TRANSFORM_DICT

EDGE_SIZE = 4
Y_K_SIZE = 6
X_K_SIZE = 6


def reduce_label(label: np.ndarray) -> Image.Image:
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return Image.fromarray(label)


def generate_edge(label: np.ndarray) -> Image.Image:
    edge = cv2.Canny(label, 0.1, 0.2)
    kernel = np.ones((EDGE_SIZE, EDGE_SIZE), np.uint8)
    # edge_pad == True
    edge = edge[Y_K_SIZE:-Y_K_SIZE, X_K_SIZE:-X_K_SIZE]
    edge = np.pad(edge, ((Y_K_SIZE, Y_K_SIZE), (X_K_SIZE, X_K_SIZE)), mode='constant')
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0
    return Image.fromarray((edge.copy() * 255).astype(np.uint8))


def transforms_check(transforms):
    names = [t.name.lower() for t in transforms]
    if 'mixing' in names:
        if names[-1] == 'mixing':
            return transforms[:-1]
        else:
            raise ValueError("Mixing transform is in the middle of transforms. This must be in the last of transforms list.")
    return transforms


def transforms_custom(conf_augmentation, training):
    phase_conf = conf_augmentation.train if training else conf_augmentation.inference

    preprocess = []
    if phase_conf:
        checked_transforms = transforms_check(phase_conf)
        for augment in checked_transforms:
            name = augment.name.lower()
            augment_kwargs = list(augment.keys())
            augment_kwargs.remove('name')
            augment_kwargs = {k:augment[k] for k in augment_kwargs}
            transform = TRANSFORM_DICT[name](**augment_kwargs)
            preprocess.append(transform)

    return TC.Compose(preprocess)


def train_transforms_pidnet(conf_augmentation, training):
    preprocess = []
    phase_conf = conf_augmentation.train if training else conf_augmentation.inference

    if phase_conf:
        checked_transforms = transforms_check(phase_conf)
        for augment in checked_transforms:
            name = augment.name.lower()
            augment_kwargs = list(augment.keys())
            augment_kwargs.remove('name')
            augment_kwargs = {k:augment[k] for k in augment_kwargs}
            transform = TRANSFORM_DICT[name](**augment_kwargs)
            preprocess.append(transform)

    return TC.Compose(preprocess, additional_targets={'edge': 'mask'})


def create_transform(model_name: str, is_training=False):
    if 'pidnet' in model_name:
        return partial(train_transforms_pidnet, training=is_training)
    return partial(transforms_custom, training=is_training)
