from typing import Optional

from torchvision.transforms.functional import InterpolationMode

from ..augmentation import custom as TC
from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def transforms_custom_train(args_augment, img_size=64):
    args = args_augment
    assert img_size > 32
    primary_tfl = [TC.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
                   TC.RandomHorizontalFlip(p=args.fliplr)
    ]
    preprocess = [
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(primary_tfl + preprocess)


def transforms_custom_eval(args_augment, img_size=64):
    args = args_augment
    assert img_size > 32
    preprocess = [
        TC.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess)


def create_transform_classification(model_name: str, is_training=False):
    return transforms_custom_train if is_training else transforms_custom_eval
