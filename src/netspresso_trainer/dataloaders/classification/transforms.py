import inspect
from typing import Optional

from torchvision.transforms.functional import InterpolationMode

from ..augmentation import custom as TC
from ..augmentation.registry import TRANSFORM_DICT
from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def transforms_custom_train(conf_augmentation):
    assert conf_augmentation.img_size > 32
    preprocess = []
    for augment in conf_augmentation.recipe:
        name = augment.name.lower()
        transform_args = list(inspect.signature(TRANSFORM_DICT[name]).parameters)
        transform_kwargs = {key:augment[key] for key in transform_args if hasattr(augment, key)}

        transform = TRANSFORM_DICT[name](**transform_kwargs)
        preprocess.append(transform)

    preprocess = preprocess + [
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess)


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
