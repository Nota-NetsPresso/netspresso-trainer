from typing import Dict, Type

from .base import BasePipeline
from .classification import ClassificationPipeline
from .segmentation import SegmentationPipeline
from .detection import DetectionPipeline

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection': DetectionPipeline
}