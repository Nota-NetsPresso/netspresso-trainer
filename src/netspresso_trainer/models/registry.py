from pathlib import Path
from typing import Callable, Dict, List, Type

import torch.nn as nn

from .backbones import cspdarknet, efficientformer, mixnet, mobilenetv3, mobilevit, resnet, segformer, vit
from .full import pidnet
from .heads.classification import fc
from .heads.detection import faster_rcnn, retinanet_head, yolox_head
from .heads.segmentation import all_mlp_decoder
from .necks import fpn, pafpn

MODEL_BACKBONE_DICT: Dict[str, Callable[..., nn.Module]] = {
    'resnet': resnet,
    'mobilenetv3': mobilenetv3,
    'segformer': segformer,
    'mobilevit': mobilevit,
    'vit': vit,
    'efficientformer': efficientformer,
    'cspdarknet': cspdarknet,
    'mixnet': mixnet,
}

MODEL_NECK_DICT: Dict[str, Callable[..., nn.Module]] = {
    'fpn': fpn,
    'pafpn': pafpn,
}

MODEL_HEAD_DICT: Dict[str, Callable[..., nn.Module]] = {
    'classification': {
        'fc': fc,
    },
    'segmentation': {
        'all_mlp_decoder': all_mlp_decoder,
    },
    'detection': {
        'faster_rcnn': faster_rcnn,
        'yolox_head': yolox_head,
        'retinanet_head': retinanet_head,
    },
}

MODEL_FULL_DICT = {
    'pidnet': pidnet
}

SUPPORTING_MODEL_LIST = list(MODEL_BACKBONE_DICT.keys()) + list(MODEL_FULL_DICT.keys())
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection']
