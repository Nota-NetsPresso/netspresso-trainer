from typing import Callable, Dict, Type

from .augmentation.transforms import create_transform
from .base import BaseCustomDataset, BaseDataSampler, BaseHFDataset
from .classification import (
    ClassficationDataSampler,
    ClassificationCustomDataset,
    ClassificationHFDataset,
)
from .detection import DetectionCustomDataset, DetectionDataSampler
from .segmentation import (
    SegmentationCustomDataset,
    SegmentationDataSampler,
    SegmentationHFDataset,
)
from .pose_estimation import (
    PoseEstimationCustomDataset,
    PoseEstimationDataSampler,
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

DATA_SAMPLER: Dict[str, Type[BaseDataSampler]] = {
    'classification': ClassficationDataSampler,
    'segmentation': SegmentationDataSampler,
    'detection': DetectionDataSampler,
    'pose_estimation': PoseEstimationDataSampler,
}
