""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019, Ross Wightman
"""
import math

import torch
from torchvision import transforms

from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from datasets.utils.augmentation.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from datasets.utils.augmentation.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy
from datasets.utils.augmentation.random_erasing import RandomErasing

# ============================== CIFAR10 =============================

def transforms_cifar10_train(
        use_prefetcher=True,
        img_size=32,
        hflip=0.5,
        vflip=0.,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010), # 0.2470, 0.2435, 0.2616 ?
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        **kwargs
):
    primary_tfl = []
    if img_size>32:
        primary_tfl += [transforms.Resize(img_size, interpolation=str_to_interp_mode('bilinear'))]
    primary_tfl += [transforms.RandomCrop(img_size, padding=4)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]
    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))
    return transforms.Compose(primary_tfl + final_tfl)


def transforms_cifar10_eval(
        use_prefetcher=True,
        img_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        **kwargs):
    tfl = []
    if img_size>32:
        tfl += [transforms.Resize(img_size, str_to_interp_mode('bilinear'))]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]
    return transforms.Compose(tfl)

# ============================== CIFAR100 =============================

def transforms_cifar100_train(
        use_prefetcher=True,
        img_size=32,
        hflip=0.5,
        vflip=0.,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        **kwargs
):
    primary_tfl = []
    if img_size>32:
        primary_tfl += [transforms.Resize(img_size, str_to_interp_mode('bilinear'))]
    primary_tfl += [transforms.RandomCrop(img_size, padding=4)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]
    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))
    return transforms.Compose(primary_tfl + final_tfl)


def transforms_cifar100_eval(
        use_prefetcher=True,
        img_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        **kwargs):
    tfl = []
    if img_size>32:
        tfl += [transforms.Resize(img_size, str_to_interp_mode('bilinear'))]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]
    return transforms.Compose(tfl)

# ============================= IMAGENET =============================

def transforms_imagenet_train(
        use_prefetcher=True,
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=None, #0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        **kwargs
):
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]

    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
        use_prefetcher=True,
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        **kwargs):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    if scale_size != img_size:
        tfl = [
            transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
            transforms.CenterCrop(img_size),
        ]
    else:
        tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]

    return transforms.Compose(tfl)


# ============================= IMAGEWOOF =============================

def transforms_imagewoof_train(
        use_prefetcher=True,
        img_size=256,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=None, #0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        **kwargs):
    return transforms_imagenet_train(use_prefetcher, img_size, scale, ratio, hflip, vflip, color_jitter, auto_augment, 
        interpolation, mean, std, re_prob, re_mode, re_count, re_num_splits, **kwargs)

def transforms_imagewoof_eval(
        use_prefetcher=True,
        img_size=256,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        **kwargs):
    return transforms_imagenet_eval(use_prefetcher, img_size, crop_pct, interpolation, mean, std, **kwargs)

# ============================= IMAGEWOOF =============================

def transforms_imagenette_train(
        use_prefetcher=True,
        img_size=256,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=None, #0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        **kwargs):
    return transforms_imagenet_train(use_prefetcher, img_size, scale, ratio, hflip, vflip, color_jitter, auto_augment, 
        interpolation, mean, std, re_prob, re_mode, re_count, re_num_splits, **kwargs)

def transforms_imagenette_eval(
        use_prefetcher=True,
        img_size=256,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        **kwargs):
    return transforms_imagenet_eval(use_prefetcher, img_size, crop_pct, interpolation, mean, std, **kwargs)

# ============================= cars196 =============================

def transforms_cars196_train(
        use_prefetcher=True,
        img_size=256,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=None, #0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        **kwargs):
    return transforms_imagenet_train(use_prefetcher, img_size, scale, ratio, hflip, vflip, color_jitter, auto_augment, 
        interpolation, mean, std, re_prob, re_mode, re_count, re_num_splits, **kwargs)

def transforms_cars196_eval(
        use_prefetcher=True,
        img_size=256,
        crop_pct=None,
        interpolation='bilinear',
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        **kwargs):
    return transforms_imagenet_eval(use_prefetcher, img_size, crop_pct, interpolation, mean, std, **kwargs)

# ================================================================================
# ================================================================================
# ================================================================================


def create_transform(dataset, img_size, is_training=False, use_prefetcher=True, **kwargs):
    if is_training:
        transform =  eval(f'transforms_{dataset}_train')(use_prefetcher, img_size, **kwargs)
    else:
        transform = eval(f'transforms_{dataset}_eval')(use_prefetcher, img_size, **kwargs)
    return transform
