from torchvision.transforms.functional import InterpolationMode

from . import custom as TC
from .registry import TRANSFORM_DICT
from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def transforms_custom_train(conf_augmentation):
    assert conf_augmentation.img_size > 32
    preprocess = []
    for augment in conf_augmentation.recipe:
        name = augment.name.lower()
        augment_kwargs = list(augment.keys())
        augment_kwargs.remove('name')
        augment_kwargs = {k:augment[k] for k in augment_kwargs}
        transform = TRANSFORM_DICT[name](**augment_kwargs)
        preprocess.append(transform)

    preprocess = preprocess + [
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess)


def transforms_custom_eval(conf_augmentation):
    assert conf_augmentation.img_size > 32
    preprocess = [
        TC.Resize((conf_augmentation.img_size, conf_augmentation.img_size), interpolation='bilinear'),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess)


def create_transform(model_name: str, is_training=False):
    return transforms_custom_train if is_training else transforms_custom_eval
