import os
import importlib
import torch
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def reduce_label(label):
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label


def train_transforms(args_augment, img_size, label, use_prefetcher):

    args = args_augment

    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    ratio_range = args.ratio0, args.ratiof
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

    train_transforms = A.Compose([
        A.Resize(int(h * scale_factor) + args.resize_add,
                 int(w * scale_factor) + args.resize_add, p=1),
        A.RandomCrop(crop_size_h, crop_size_w),
        A.Flip(p=args.flipp),
        A.ColorJitter(brightness=args.brightness,
                      contrast=args.contrast,
                      saturation=args.saturation,
                      hue=args.hue,
                      p=args.colorjitter_p),
        A.Normalize(mean=args.norm_mean, std=args.norm_std),
        A.PadIfNeeded(crop_size_h, crop_size_w,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        ToTensorV2()
    ])

    return train_transforms


def val_transforms(args_augment, img_size, label, use_prefetcher):

    args = args_augment

    if args.reduce_zero_label == True:
        label = reduce_label(label)

    h, w = img_size[:2]
    scale_factor = min(args.max_scale / max(h, w), args.min_scale / min(h, w))
    val_transforms = A.Compose([
        A.Resize(int(h * scale_factor), int(w * scale_factor), p=1),
        A.Normalize(mean=args.norm_mean, std=args.norm_std),
        ToTensorV2()
    ])

    return val_transforms


def infer_transforms(args_augment, img_size):

    args = args_augment

    h, w = img_size[:2]
    scale_factor = min(args.max_scale / max(h, w), args.min_scale / min(h, w))

    val_transforms = A.Compose([
        A.Resize(int(h * scale_factor), int(w * scale_factor), p=1),
        A.Normalize(mean=args.norm_mean, std=args.norm_std),
        ToTensorV2()
    ])

    return val_transforms


def create_segmentation_transform(args, img_size, label, is_training=False, use_prefetcher=True):

    if is_training:
        transform = train_transforms(args, img_size, label, use_prefetcher)
    else:
        transform = val_transforms(args, img_size, label, use_prefetcher)
    return transform
