from pathlib import Path
from typing import Callable, Dict, List, Type

import torch.nn as nn

from .backbones import cspdarknet, efficientformer, mixnet, mixtransformer, mobilenetv3, mobilevit, resnet, vit
from .base import ClassificationModel, DetectionModel, SegmentationModel, TaskModel
from .full import pidnet
from .heads.classification import fc
from .heads.detection import anchor_decoupled_head, anchor_free_decoupled_head
from .heads.segmentation import all_mlp_decoder
from .necks import fpn, yolopafpn

MODEL_BACKBONE_DICT: Dict[str, Callable[..., nn.Module]] = {
    'resnet': resnet,
    'mobilenetv3': mobilenetv3,
    'mixtransformer': mixtransformer,
    'mobilevit': mobilevit,
    'vit': vit,
    'efficientformer': efficientformer,
    'cspdarknet': cspdarknet,
    'mixnet': mixnet,
}

MODEL_NECK_DICT: Dict[str, Callable[..., nn.Module]] = {
    'fpn': fpn,
    'yolopafpn': yolopafpn,
}

MODEL_HEAD_DICT: Dict[str, Callable[..., nn.Module]] = {
    'classification': {
        'fc': fc,
    },
    'segmentation': {
        'all_mlp_decoder': all_mlp_decoder,
    },
    'detection': {
        'anchor_free_decoupled_head': anchor_free_decoupled_head,
        'anchor_decoupled_head': anchor_decoupled_head,
    },
}

MODEL_FULL_DICT = {
    'pidnet': pidnet
}

SUPPORTING_MODEL_LIST = list(MODEL_BACKBONE_DICT.keys()) + list(MODEL_FULL_DICT.keys())
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection']

TASK_MODEL_DICT: Dict[str, Type[TaskModel]] = {
    'classification': ClassificationModel,
    'segmentation': SegmentationModel,
    'detection': DetectionModel
}
