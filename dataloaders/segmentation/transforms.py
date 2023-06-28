from typing import Optional

import numpy as np
import cv2
import PIL.Image as Image

import dataloaders.augmentation.custom as TC
from dataloaders.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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


def train_transforms_segformer(args_augment):

    args = args_augment

    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    scale_ratio = (args.resize_ratio0, args.resize_ratiof)
    
    train_transforms_composed = TC.Compose([
        TC.RandomResizedCrop((crop_size_h, crop_size_w), scale=scale_ratio, ratio=(1.0, 1.0)),
        TC.RandomHorizontalFlip(p=args.fliplr),
        TC.ColorJitter(brightness=args.color_jitter.brightness,
                       contrast=args.color_jitter.contrast,
                       saturation=args.color_jitter.saturation,
                       hue=args.color_jitter.hue,
                       p=args.color_jitter.colorjitter_p),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return train_transforms_composed

def val_transforms_segformer(args_augment):

    args = args_augment
    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    val_transforms_composed = TC.Compose([
        TC.Resize((crop_size_h, crop_size_w)),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return val_transforms_composed


def infer_transforms_segformer(args_augment):
    return


def train_transforms_pidnet(args_augment):
    args = args_augment

    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    scale_ratio = (args.resize_ratio0, args.resize_ratiof)

    train_transforms_composed = TC.Compose(
        [
            TC.RandomResizedCrop((crop_size_h, crop_size_w), scale=scale_ratio, ratio=(1.0, 1.0)),
            TC.RandomHorizontalFlip(p=args.fliplr),
            TC.ToTensor(),
            TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ],
        additional_targets={'edge': 'mask'}
    )
    
    return train_transforms_composed


def val_transforms_pidnet(args_augment):
    args = args_augment
    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    val_transforms_composed = TC.Compose(
        [
            TC.Resize((crop_size_h, crop_size_w)),
            TC.ToTensor(),
            TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ],
        additional_targets={'edge': 'mask'}
    )

    return val_transforms_composed


def infer_transforms_pidnet(args_augment):
    return


def train_transforms_efficientformer(args_augment):

    args = args_augment

    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    scale_ratio = (args.resize_ratio0, args.resize_ratiof)
    
    train_transforms_composed = TC.Compose([
        TC.RandomResizedCrop((crop_size_h, crop_size_w), scale=scale_ratio, ratio=(1.0, 1.0)),
        TC.RandomHorizontalFlip(p=args.fliplr),
        TC.ColorJitter(brightness=args.color_jitter.brightness,
                       contrast=args.color_jitter.contrast,
                       saturation=args.color_jitter.saturation,
                       hue=args.color_jitter.hue,
                       p=args.color_jitter.colorjitter_p),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return train_transforms_composed

def val_transforms_efficientformer(args_augment):

    args = args_augment
    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    val_transforms_composed = TC.Compose([
        TC.Resize((crop_size_h, crop_size_w)),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    return val_transforms_composed

def create_segmentation_transform(args, is_training=False):

    if 'segformer' in args.train.architecture.values():
        if is_training:
            return train_transforms_segformer
        return val_transforms_segformer
    elif 'pidnet' in args.train.architecture.values():
        if is_training:
            return train_transforms_pidnet
        return val_transforms_pidnet
    elif 'efficientformer' in args.train.architecture.values():
        if is_training:
            return train_transforms_efficientformer
        return val_transforms_efficientformer
    raise ValueError(f"No such model named: {args.train.architecture.values()} !!!")
