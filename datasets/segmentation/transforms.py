import os
import importlib
import random

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

AUG_FLIP_PROP = 0.5
EDGE_SIZE = 4
Y_K_SIZE = 6
X_K_SIZE = 6

def reduce_label(label):
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label

def generate_edge(label):
    edge = cv2.Canny(label, 0.1, 0.2)
    kernel = np.ones((EDGE_SIZE, EDGE_SIZE), np.uint8)
    # edge_pad == True
    edge = edge[Y_K_SIZE:-Y_K_SIZE, X_K_SIZE:-X_K_SIZE]
    edge = np.pad(edge, ((Y_K_SIZE, Y_K_SIZE), (X_K_SIZE, X_K_SIZE)), mode='constant')
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

    return edge.astype(np.float32).copy()
def resize_and_crop(args_augment, image, label):

    edge = generate_edge(label)
    
    f_scale = np.random.random_sample() * (args_augment.resize_ratiof - args_augment.resize_ratio0) + args_augment.resize_add 

    image = cv2.resize(image, dsize=None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, dsize=None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
    edge = cv2.resize(edge, dsize=None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

    # random crop augmentation
    img_h, img_w = image.shape[:2]
    h_off = random.randint(0, img_h - args_augment.crop_size_h)
    w_off = random.randint(0, img_w - args_augment.crop_size_w)

    image = image[h_off: h_off + args_augment.crop_size_h,
                    w_off: w_off + args_augment.crop_size_w, :]  # H x W x C(=3)
    label = label[h_off: h_off + args_augment.crop_size_h,
                    w_off: w_off + args_augment.crop_size_w]  # H x W
    edge = edge[h_off: h_off + args_augment.crop_size_h,
                w_off: w_off + args_augment.crop_size_w]  # H x W

    return image, label, edge


def train_transforms_segformer(args_augment, img_size, label, use_prefetcher):

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


def val_transforms_segformer(args_augment, img_size, label, use_prefetcher):

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


def infer_transforms_segformer(args_augment, img_size):
    return
    args = args_augment

    h, w = img_size[:2]
    scale_factor = min(args.max_scale / max(h, w), args.min_scale / min(h, w))

    val_transforms_composed = A.Compose([
        A.Resize(int(h * scale_factor), int(w * scale_factor), p=1),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ])

    return val_transforms_composed

def train_transforms_pidnet(args_augment, img_size, label, use_prefetcher):
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

    train_transforms_composed = A.Compose(
        [
            A.Resize(int(h * scale_factor) + args.resize_add,
                    int(w * scale_factor) + args.resize_add, p=1),
            A.RandomCrop(crop_size_h, crop_size_w),
            A.Flip(p=args.fliplr),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            A.PadIfNeeded(crop_size_h, crop_size_w,
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
            ToTensorV2()
        ],
        additional_targets={'edge': 'mask'})
    
    return train_transforms_composed

def val_transforms_pidnet(args_augment, img_size, label, use_prefetcher):
    args = args_augment
    
    crop_size_h = args.crop_size_h
    crop_size_w = args.crop_size_w

    if args.reduce_zero_label == True:
        label = reduce_label(label)

    h, w = img_size[:2]
    scale_factor = min(args.max_scale / max(h, w), args.min_scale / min(h, w))
    val_transforms_composed = A.Compose([
        A.Resize(crop_size_h, crop_size_w),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ])

    return val_transforms_composed


def infer_transforms_pidnet(args_augment, img_size):
    return
    def compose_transform(image):
        out = {}
        image = np.asarray(image, np.float32)
        image -= args_augment.mean
        
        image = cv2.resize(image, dsize=(args_augment.crop_size.h, args_augment.crop_size.w), interpolation=cv2.INTER_LINEAR)
        image = image.transpose((2, 0, 1))  # C x H x W
        
        out = {
            'image': image,
        }
        
        return out
    
    return compose_transform

def create_segmentation_transform(args, is_training=False):

    if 'segformer' in args.train.architecture.values():
        if is_training:
            transform = train_transforms_segformer
        else:
            transform = val_transforms_segformer
    elif 'pidnet' in args.train.architecture.values():
        if is_training:
            transform = train_transforms_pidnet
        else:
            transform = val_transforms_pidnet
    return transform
