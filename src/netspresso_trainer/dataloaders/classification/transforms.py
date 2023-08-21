from typing import Optional

from torchvision.transforms.functional import InterpolationMode

from ..augmentation import custom as TC
from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def transforms_custom_train(conf_augmentation):
    assert conf_augmentation.img_size > 32
    primary_tfl = [TC.RandomResizedCrop(conf_augmentation.img_size, interpolation=InterpolationMode.BILINEAR),
                   TC.RandomHorizontalFlip(p=conf_augmentation.fliplr)
                   ]
    preprocess = [
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(primary_tfl + preprocess)


def transforms_custom_eval(conf_augmentation):
    assert conf_augmentation.img_size > 32
    preprocess = [
        TC.Resize((conf_augmentation.img_size, conf_augmentation.img_size),
                  interpolation=InterpolationMode.BILINEAR),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess)


def create_transform_classification(model_name: str, is_training=False):
    return transforms_custom_train if is_training else transforms_custom_eval
