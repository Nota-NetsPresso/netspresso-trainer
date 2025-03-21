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

from typing import Dict, Type

from .classification import ClassificationPostprocessor
from .detection import DetectionPostprocessor
from .pose_estimation import PoseEstimationPostprocessor
from .segmentation import SegmentationPostprocessor

POSTPROCESSOR_DICT = {
    'fc': ClassificationPostprocessor,
    'fc_conv': ClassificationPostprocessor,
    'all_mlp_decoder': SegmentationPostprocessor,
    'anchor_free_decoupled_head': DetectionPostprocessor,
    'pidnet': SegmentationPostprocessor,
    'anchor_decoupled_head': DetectionPostprocessor,
    'yolo_fastest_head_v2': DetectionPostprocessor,
    'yolo_detection_head': DetectionPostprocessor,
    'rtmcc': PoseEstimationPostprocessor,
    'rtdetr_head': DetectionPostprocessor,
}
