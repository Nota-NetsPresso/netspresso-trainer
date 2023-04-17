import os
import importlib

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def reduce_label(label):
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label


def train_transforms(args_augment, img_size, label, use_prefetcher):

    args = args_augment

    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    ratio_range = args.resize_ratio0, args.resize_ratiof
    img_scale = args.max_scale, args.min_scale

    if args.reduce_zero_label:
        label = reduce_label(label)

    h, w = img_size[:2]
    ratio = np.random.random_sample() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
    scale = (img_scale[0] * ratio, img_scale[1] * ratio)
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    if (scale_factor * min(h, w)) < min(crop_size_h, crop_size_w):
        scale_factor = min(crop_size_h, crop_size_w) / min(h, w)

    train_transforms_composed = A.Compose([
        A.Resize(int(h * scale_factor) + args.resize_add,
                 int(w * scale_factor) + args.resize_add, p=1),
        A.RandomCrop(crop_size_h, crop_size_w),
        A.Flip(p=args.fliplr),
        A.ColorJitter(brightness=args.color_jitter.brightness,
                      contrast=args.color_jitter.contrast,
                      saturation=args.color_jitter.saturation,
                      hue=args.color_jitter.hue,
                      p=args.color_jitter.colorjitter_p),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        A.PadIfNeeded(crop_size_h, crop_size_w,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        ToTensorV2()
    ])

    return train_transforms_composed


def val_transforms(args_augment, img_size, label, use_prefetcher):

    args = args_augment

    if args.reduce_zero_label == True:
        label = reduce_label(label)

    h, w = img_size[:2]
    scale_factor = min(args.max_scale / max(h, w), args.min_scale / min(h, w))
    val_transforms_composed = A.Compose([
        A.Resize(int(h * scale_factor), int(w * scale_factor), p=1),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ])

    return val_transforms_composed


def infer_transforms(args_augment, img_size):

    args = args_augment

    h, w = img_size[:2]
    scale_factor = min(args.max_scale / max(h, w), args.min_scale / min(h, w))

    val_transforms_composed = A.Compose([
        A.Resize(int(h * scale_factor), int(w * scale_factor), p=1),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ])

    return val_transforms_composed


def create_segmentation_transform(is_training=False):

    if is_training:
        transform = train_transforms
    else:
        transform = val_transforms
    return transform
