# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from pathlib import Path
from typing import Callable, Dict, List, Type

import torch.nn as nn

from .backbones import (
    cspdarknet,
    efficientformer,
    mixnet,
    mixtransformer,
    mobilenetv3,
    mobilenetv4,
    mobilevit,
    resnet,
    shufflenetv2,
    vit,
)
from .base import ClassificationModel, DetectionModel, PoseEstimationModel, SegmentationModel, TaskModel
from .full import pidnet
from .heads.classification import fc, fc_conv
from .heads.detection import anchor_decoupled_head, anchor_free_decoupled_head, rtdetr_head, yolo_fastest_head_v2
from .heads.pose_estimation import rtmcc
from .heads.segmentation import all_mlp_decoder
from .necks import fpn, lightfpn, rtdetr_hybrid_encoder, yolopafpn

MODEL_BACKBONE_DICT: Dict[str, Callable[..., nn.Module]] = {
    'resnet': resnet,
    'mobilenetv3': mobilenetv3,
    'mobilenetv4': mobilenetv4,
    'mixtransformer': mixtransformer,
    'mobilevit': mobilevit,
    'vit': vit,
    'efficientformer': efficientformer,
    'cspdarknet': cspdarknet,
    'shufflenetv2': shufflenetv2,
    'mixnet': mixnet,
}

MODEL_NECK_DICT: Dict[str, Callable[..., nn.Module]] = {
    'fpn': fpn,
    'lightfpn': lightfpn,
    'yolopafpn': yolopafpn,
    'rtdetr_hybrid_encoder': rtdetr_hybrid_encoder,
}

MODEL_HEAD_DICT: Dict[str, Callable[..., nn.Module]] = {
    'classification': {
        'fc': fc,
        'fc_conv': fc_conv,
    },
    'segmentation': {
        'all_mlp_decoder': all_mlp_decoder,
    },
    'detection': {
        'anchor_free_decoupled_head': anchor_free_decoupled_head,
        'anchor_decoupled_head': anchor_decoupled_head,
        'yolo_fastest_head_v2': yolo_fastest_head_v2,
        'rtdetr_head': rtdetr_head
    },
    'pose_estimation': {
        'rtmcc': rtmcc,
    }
}

MODEL_FULL_DICT = {
    'pidnet': pidnet
}

SUPPORTING_MODEL_LIST = list(MODEL_BACKBONE_DICT.keys()) + list(MODEL_FULL_DICT.keys())
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection', 'pose_estimation']

TASK_MODEL_DICT: Dict[str, Type[TaskModel]] = {
    'classification': ClassificationModel,
    'segmentation': SegmentationModel,
    'detection': DetectionModel,
    'pose_estimation': PoseEstimationModel,
}
