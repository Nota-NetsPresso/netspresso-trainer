from typing import Dict, Type, Callable

from .base import BaseCustomDataset, BaseHFDataset, BaseDataSampler
from .classification import (
    ClassificationCustomDataset, ClassificationHFDataset, ClassficationDataSampler, create_transform_classification
)
from .segmentation import (
    SegmentationCustomDataset, SegmentationHFDataset, SegmentationDataSampler, create_transform_segmentation
)
from .detection import (
    DetectionCustomDataset, DetectionDataSampler, create_transform_detection
)

from .augmentation import custom as TC

CREATE_TRANSFORM: Dict[str, Callable[..., Callable[..., TC.Compose]]] = {
    'classification': create_transform_classification,
    'segmentation': create_transform_segmentation,
    'detection': create_transform_detection
}

CUSTOM_DATASET: Dict[str, Type[BaseCustomDataset]] = {
    'classification': ClassificationCustomDataset,
    'segmentation': SegmentationCustomDataset,
    'detection': DetectionCustomDataset
}

HUGGINGFACE_DATASET: Dict[str, Type[BaseHFDataset]] = {
    'classification': ClassificationHFDataset,
    'segmentation': SegmentationHFDataset
}

DATA_SAMPLER: Dict[str, Type[BaseDataSampler]] = {
    'classification': ClassficationDataSampler,
    'segmentation': SegmentationDataSampler,
    'detection': DetectionDataSampler
}