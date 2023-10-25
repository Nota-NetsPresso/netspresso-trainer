from typing import Dict, Type, List

from .base import BasePipeline
from .classification import ClassificationPipeline
from .detection import TwoStageDetectionPipeline
from .segmentation import SegmentationPipeline

SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection']

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection-two-stage': TwoStageDetectionPipeline
}