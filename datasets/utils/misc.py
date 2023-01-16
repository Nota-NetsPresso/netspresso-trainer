from itertools import repeat

from datasets.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


def transforms_config(dataset: str, train: bool, cfg = None):
    transf_config = {}

    if train:
        transf_config = {
            'hflip': 0.5,
            'vflip': 0,
            'mean': IMAGENET_DEFAULT_MEAN, 
            'std': IMAGENET_DEFAULT_STD
        }
    else:
        transf_config = {
            'mean': IMAGENET_DEFAULT_MEAN, 
            'std': IMAGENET_DEFAULT_STD
        }

    if cfg is not None:
        for key in cfg.keys():
            transf_config[key] = cfg[key]

    return transf_config