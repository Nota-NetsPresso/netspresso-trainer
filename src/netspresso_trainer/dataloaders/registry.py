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

from typing import Callable, Dict, Type

from .augmentation.transforms import create_transform
from .base import BaseCustomDataset, BaseHFDataset, BaseSampleLoader
from .classification import (
    ClassficationSampleLoader,
    ClassificationCustomDataset,
    ClassificationHFDataset,
)
from .detection import DetectionCustomDataset, DetectionSampleLoader
from .pose_estimation import (
    PoseEstimationCustomDataset,
    PoseEstimationSampleLoader,
)
from .segmentation import (
    SegmentationCustomDataset,
    SegmentationHFDataset,
    SegmentationSampleLoader,
)

CREATE_TRANSFORM = create_transform

CUSTOM_DATASET: Dict[str, Type[BaseCustomDataset]] = {
    'classification': ClassificationCustomDataset,
    'segmentation': SegmentationCustomDataset,
    'detection': DetectionCustomDataset,
    'pose_estimation': PoseEstimationCustomDataset,
}

HUGGINGFACE_DATASET: Dict[str, Type[BaseHFDataset]] = {
    'classification': ClassificationHFDataset,
    'segmentation': SegmentationHFDataset
}

DATA_SAMPLER: Dict[str, Type[BaseSampleLoader]] = {
    'classification': ClassficationSampleLoader,
    'segmentation': SegmentationSampleLoader,
    'detection': DetectionSampleLoader,
    'pose_estimation': PoseEstimationSampleLoader,
}
