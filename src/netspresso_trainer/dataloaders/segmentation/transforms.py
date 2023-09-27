from typing import Optional

import cv2
import numpy as np
import PIL.Image as Image

from ..augmentation import custom as TC
from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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


def train_transforms_segmentation(conf_augmentation):

    crop_size_h = conf_augmentation.crop_size_h
    crop_size_w = conf_augmentation.crop_size_w

    scale_ratio = (conf_augmentation.resize_ratio0, conf_augmentation.resize_ratiof)
    
    train_transforms_composed = TC.Compose([
        TC.RandomResizedCrop((crop_size_h, crop_size_w), scale=scale_ratio, ratio=(1.0, 1.0)),
        TC.RandomHorizontalFlip(p=conf_augmentation.fliplr),
        TC.ColorJitter(brightness=conf_augmentation.color_jitter.brightness,
                       contrast=conf_augmentation.color_jitter.contrast,
                       saturation=conf_augmentation.color_jitter.saturation,
                       hue=conf_augmentation.color_jitter.hue,
                       p=conf_augmentation.color_jitter.colorjitter_p),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return train_transforms_composed

def val_transforms_segmentation(conf_augmentation):

    crop_size_h = conf_augmentation.crop_size_h
    crop_size_w = conf_augmentation.crop_size_w

    val_transforms_composed = TC.Compose([
        TC.Resize((crop_size_h, crop_size_w)),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return val_transforms_composed


def infer_transforms_segmentation(conf_augmentation):
    return


def train_transforms_pidnet(conf_augmentation):

    crop_size_h = conf_augmentation.crop_size_h
    crop_size_w = conf_augmentation.crop_size_w

    scale_ratio = (conf_augmentation.resize_ratio0, conf_augmentation.resize_ratiof)

    train_transforms_composed = TC.Compose(
        [
            TC.RandomResizedCrop((crop_size_h, crop_size_w), scale=scale_ratio, ratio=(1.0, 1.0)),
            TC.RandomHorizontalFlip(p=conf_augmentation.fliplr),
            TC.ToTensor(),
            TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ],
        additional_targets={'edge': 'mask'}
    )
    
    return train_transforms_composed


def val_transforms_pidnet(conf_augmentation):

    crop_size_h = conf_augmentation.crop_size_h
    crop_size_w = conf_augmentation.crop_size_w

    val_transforms_composed = TC.Compose(
        [
            TC.Resize((crop_size_h, crop_size_w)),
            TC.ToTensor(),
            TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ],
        additional_targets={'edge': 'mask'}
    )

    return val_transforms_composed


def infer_transforms_pidnet(conf_augmentation):
    return


def create_transform_segmentation(model_name: str, is_training=False):

    if model_name == 'pidnet':
        if is_training:
            return train_transforms_pidnet
        return val_transforms_pidnet
    if is_training:
        return train_transforms_segmentation
    return val_transforms_segmentation
