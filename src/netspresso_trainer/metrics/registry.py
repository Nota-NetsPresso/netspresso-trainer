from typing import Callable, Dict, Literal, Type

from .base import BaseMetric
from .classification import ClassificationMetric
from .detection import DetectionMetric
from .pose_estimation import PoseEstimationMetric
from .segmentation import SegmentationMetric

TASK_METRIC: Dict[Literal['classification', 'segmentation', 'detection'], Type[BaseMetric]] = {
    'classification': ClassificationMetric,
    'segmentation': SegmentationMetric,
    'detection': DetectionMetric,
    'pose_estimation': PoseEstimationMetric,
}

PHASE_LIST = ['train', 'valid', 'test']
