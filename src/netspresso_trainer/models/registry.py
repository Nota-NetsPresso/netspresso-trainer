from typing import List, Dict, Type, Callable
from pathlib import Path

import torch.nn as nn

from .backbones import resnet50, segformer, mobilevit, vit, efficientformer
from .full import pidnet

from .heads.classification import fc
from .heads.segmentation import all_mlp_decoder
from .heads.detection import efficientformer_detection_head

MODEL_BACKBONE_DICT: Dict[str, Callable[..., nn.Module]] = {
    'resnet50': resnet50,
    'segformer': segformer,
    'mobilevit': mobilevit,
    'vit': vit,
    'efficientformer': efficientformer,
}

MODEL_HEAD_DICT: Dict[str, Callable[..., nn.Module]] = {
    'classification': {
        'fc': fc,
    },
    'segmentation': {
        'all_mlp_decoder': all_mlp_decoder,
    },
    'detection': {
        'efficientformer_detection_head': efficientformer_detection_head
    },
}

MODEL_FULL_DICT = {
    'pidnet': pidnet
}

SUPPORTING_MODEL_LIST = list(MODEL_BACKBONE_DICT.keys()) + list(MODEL_FULL_DICT.keys())
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection']
