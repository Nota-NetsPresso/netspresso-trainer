from typing import List, Dict, Type, Callable
from pathlib import Path

import torch.nn as nn

from models.backbones.experimental.resnet import resnet50
from models.backbones.experimental.segformer import segformer
from models.backbones.experimental.mobilevit import mobilevit
from models.backbones.experimental.vit import vit
from models.backbones.experimental.efficientformer import efficientformer
from models.full.experimental.pidnet import pidnet

from models.heads.classification.experimental.fc import fc
from models.heads.segmentation.experimental.decode_head import segformer_decode_head, efficientformer_decode_head
from models.heads.detection.experimental.basic import efficientformer_detection_head

MODEL_BACKBONE_DICT: Dict[str, Callable[..., nn.Module]] = {
    'resnet50': resnet50,
    'segformer': segformer,
    'mobilevit': mobilevit,
    'vit': vit,
    'efficientformer': efficientformer,
    'pidnet': pidnet
}

MODEL_HEAD_DICT: Dict[str, Callable[..., nn.Module]] = {
    'classification': {
        'fc': fc,
    },
    'segmentation': {
        'segformer_decode_head': segformer_decode_head,
        'efficientformer_decode_head': efficientformer_decode_head,
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