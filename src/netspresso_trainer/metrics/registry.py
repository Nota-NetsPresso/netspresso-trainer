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

from typing import Callable, Dict, Literal, Type

from .base import BaseMetric
from .classification import Accuracy
from .detection import mAP
from .pose_estimation import PoseEstimationMetric
from .segmentation import mIoU

METRIC_LIST: Dict[str, Type[BaseMetric]] = {
    'accuracy': Accuracy,
    'miou': mIoU,
    'map': mAP,
    'pose_estimation': PoseEstimationMetric,
}

PHASE_LIST = ['train', 'valid', 'test']

TASK_AVAILABLE_METRICS = {
    'classification': ['accuracy'],
    'segmentation': ['miou'],
    'detection': ['map'],
}

TASK_DEFUALT_METRICS = {
    'classification': ['accuracy'],
    'segmentation': ['miou'],
    'detection': ['map'],
}