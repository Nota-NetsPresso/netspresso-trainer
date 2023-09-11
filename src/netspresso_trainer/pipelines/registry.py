from typing import Dict, Type

from .base import BasePipeline
from .classification import ClassificationPipeline
from .detection import DetectionPipeline
from .segmentation import SegmentationPipeline

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection': DetectionPipeline
}