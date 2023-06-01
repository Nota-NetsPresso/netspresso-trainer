from typing import Optional

import numpy as np
import cv2
import PIL.Image as Image

import datasets.augmentation.custom as TC
from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def train_transforms_efficientformer(args_augment):

    args = args_augment
    
    train_transforms_composed = TC.Compose([
        TC.Identity(),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return train_transforms_composed

def val_transforms_efficientformer(args_augment):

    args = args_augment
    
    val_transforms_composed = TC.Compose([
        TC.Identity(),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return val_transforms_composed

def create_detection_transform(args, is_training=False):

    if 'efficientformer' in args.train.architecture.values():
        if is_training:
            return train_transforms_efficientformer
        return val_transforms_efficientformer
    raise ValueError(f"No such model named: {args.train.architecture.values()} !!!")
