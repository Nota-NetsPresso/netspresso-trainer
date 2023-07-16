from typing import Dict, Type

from pipelines.base import BasePipeline
from pipelines.classification import ClassificationPipeline
from pipelines.segmentation import SegmentationPipeline
from pipelines.detection import DetectionPipeline

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection': DetectionPipeline
}