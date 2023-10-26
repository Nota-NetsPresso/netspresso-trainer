from typing import Dict, List, Type

from .base import BasePipeline
from .classification import ClassificationPipeline
from .detection import TwoStageDetectionPipeline
from .segmentation import SegmentationPipeline

# TODO: Temporary defined. It should be integrated with `..models.registry.SUPPORTING_TASK_LIST`
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection']

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection-two-stage': TwoStageDetectionPipeline
}