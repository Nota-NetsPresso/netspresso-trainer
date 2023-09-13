from typing import Callable, Dict, Type

from .augmentation import custom as TC
from .base import BaseCustomDataset, BaseDataSampler, BaseHFDataset
from .classification import (
    ClassficationDataSampler,
    ClassificationCustomDataset,
    ClassificationHFDataset,
    create_transform_classification,
)
from .detection import DetectionCustomDataset, DetectionDataSampler, create_transform_detection
from .segmentation import (
    SegmentationCustomDataset,
    SegmentationDataSampler,
    SegmentationHFDataset,
    create_transform_segmentation,
)

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