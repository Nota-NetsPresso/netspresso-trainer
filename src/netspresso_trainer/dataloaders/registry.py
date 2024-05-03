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
