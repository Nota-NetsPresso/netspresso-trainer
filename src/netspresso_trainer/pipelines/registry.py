from typing import Dict, List, Type

from .base import BasePipeline
from .classification import ClassificationPipeline
from .detection import DetectionPipeline
from .segmentation import SegmentationPipeline
from .pose_estimation import PoseEstimationPipeline

# TODO: Temporary defined. It should be integrated with `..models.registry.SUPPORTING_TASK_LIST`
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection', 'pose_estimation']

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection': DetectionPipeline,
    'pose_estimation': PoseEstimationPipeline,
}
