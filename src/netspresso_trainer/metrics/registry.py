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
from .classification import Top1Accuracy, Top5Accuracy
from .detection import mAP50, mAP75, mAP50_95
from .pose_estimation import PoseEstimationMetric
from .segmentation import mIoU, PixelAccuracy

METRIC_LIST: Dict[str, Type[BaseMetric]] = {
    'top1_accuracy': Top1Accuracy,
    'top5_accuracy': Top5Accuracy,
    'miou': mIoU,
    'pixel_accuracy': PixelAccuracy,
    'map50': mAP50,
    'map75': mAP75,
    'map50_95': mAP50_95,
    'pose_estimation': PoseEstimationMetric,
}

PHASE_LIST = ['train', 'valid', 'test']

TASK_AVAILABLE_METRICS = {
    'classification': ['top1_accuracy', 'top5_accuracy'],
    'segmentation': ['miou', 'pixel_accuracy'],
    'detection': ['map50', 'map75', 'map50_95'],
}

TASK_DEFUALT_METRICS = {
    'classification': ['top1_accuracy', 'top5_accuracy'],
    'segmentation': ['miou'],
    'detection': ['map50', 'map75', 'map50_95'],
}