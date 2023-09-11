from typing import Optional

import cv2
import numpy as np
import PIL.Image as Image

from ..augmentation import custom as TC
from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def train_transforms_efficientformer(conf_augmentation):
    
    crop_size_h = conf_augmentation.crop_size_h
    crop_size_w = conf_augmentation.crop_size_w
    
    train_transforms_composed = TC.Compose([
        TC.Resize(size=(crop_size_h, crop_size_w)),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return train_transforms_composed

def val_transforms_efficientformer(conf_augmentation):
    
    crop_size_h = conf_augmentation.crop_size_h
    crop_size_w = conf_augmentation.crop_size_w
    
    val_transforms_composed = TC.Compose([
        TC.Resize(size=(crop_size_h, crop_size_w)),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return val_transforms_composed

def create_transform_detection(model_name: str, is_training=False):

    if model_name == 'efficientformer':
        if is_training:
            return train_transforms_efficientformer
        return val_transforms_efficientformer
    raise ValueError(f"No such model named: {model_name} !!!")
