import torch
from torchvision import transforms

from datasets.utils.augmentation.transforms import str_to_interp_mode, ToNumpy
from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def transforms_custom_train(
        use_prefetcher=True,
        img_size=32,
        hflip=0.5,
        vflip=0.,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
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
    return transforms.Compose(primary_tfl + final_tfl)


def transforms_custom_eval(
        use_prefetcher=True,
        img_size=32,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        **kwargs):
    tfl = []
    if img_size>32:
        tfl += [transforms.Resize((img_size, img_size), str_to_interp_mode('bilinear'))]
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


# ================================================================================
# ================================================================================
# ================================================================================


def create_classification_transform(dataset, img_size, is_training=False, use_prefetcher=True):
    
    if is_training:
        transform =  transforms_custom_train(use_prefetcher, img_size)
    else:
        transform = transforms_custom_eval(use_prefetcher, img_size)
    return transform
