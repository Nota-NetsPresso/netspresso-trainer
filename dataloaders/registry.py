from typing import Dict, Type

from dataloaders.base import BaseCustomDataset, BaseHFDataset, BaseDataSampler
from dataloaders.classification import (
    ClassificationCustomDataset, ClassificationHFDataset, ClassficationDataSampler, create_classification_transform
)
from dataloaders.segmentation import (
    SegmentationCustomDataset, SegmentationHFDataset, SegmentationDataSampler, create_segmentation_transform
)
from dataloaders.detection import (
    DetectionCustomDataset, DetectionDataSampler, create_detection_transform, detection_collate_fn
)


CREATE_TRANSFORM = {
    'classification': create_classification_transform,
    'segmentation': create_segmentation_transform,
    'detection': create_detection_transform
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