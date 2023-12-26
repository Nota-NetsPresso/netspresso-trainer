import cv2
import numpy as np
import PIL.Image as Image

from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from . import custom as TC
from .registry import TRANSFORM_DICT

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


def transforms_custom_train(conf_augmentation):
    assert conf_augmentation.img_size > 32
    preprocess = []
    for augment in conf_augmentation.transforms:
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
        TC.Resize((conf_augmentation.img_size, conf_augmentation.img_size), interpolation='bilinear', max_size=None),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess)


def train_transforms_pidnet(conf_augmentation):
    preprocess = []
    for augment in conf_augmentation.transforms:
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
    return TC.Compose(preprocess, additional_targets={'edge': 'mask'})


def val_transforms_pidnet(conf_augmentation):
    assert conf_augmentation.img_size > 32
    preprocess = [
        TC.Resize((conf_augmentation.img_size, conf_augmentation.img_size), interpolation='bilinear', max_size=None),
        TC.ToTensor(),
        TC.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return TC.Compose(preprocess, additional_targets={'edge': 'mask'})


def create_transform(model_name: str, is_training=False):
    if 'pidnet' in model_name:
        return train_transforms_pidnet if is_training else val_transforms_pidnet
    return transforms_custom_train if is_training else transforms_custom_eval
