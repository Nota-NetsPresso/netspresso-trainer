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

from .common import CrossEntropyLoss, SigmoidFocalLoss
from .detection import DETRLoss, RetinaNetLoss, YOLOXLoss
from .pose_estimation import RTMCCLoss
from .segmentation import PIDNetLoss, SegCrossEntropyLoss

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'seg_cross_entropy': SegCrossEntropyLoss,
    'pidnet_loss': PIDNetLoss,
    'yolox_loss': YOLOXLoss,
    'retinanet_loss': RetinaNetLoss,
    'detr_loss': DETRLoss,
    'focal_loss': SigmoidFocalLoss,
    'rtmcc_loss': RTMCCLoss,
}

PHASE_LIST = ['train', 'valid', 'test']
