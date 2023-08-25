from typing import Dict, Callable, Literal, Type

from .base import BaseMetric
from .classification import ClassificationMetric
from .segmentation import SegmentationMetric
from .detection import DetectionMetric


TASK_METRIC: Dict[Literal['classification', 'segmentation', 'detection'], Type[BaseMetric]] = {
    'classification': ClassificationMetric,
    'segmentation': SegmentationMetric,
    'detection': DetectionMetric
}
