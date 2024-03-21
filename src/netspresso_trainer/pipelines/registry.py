from typing import Dict, List, Type

from .task_pipelines.base import BasePipeline
from .task_pipelines.classification import ClassificationPipeline
from .task_pipelines.detection import DetectionPipeline
from .task_pipelines.pose_estimation import PoseEstimationPipeline
from .task_pipelines.segmentation import SegmentationPipeline

# TODO: Temporary defined. It should be integrated with `..models.registry.SUPPORTING_TASK_LIST`
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection', 'pose_estimation']

TASK_PIPELINE: Dict[str, Type[BasePipeline]]= {
    'classification': ClassificationPipeline,
    'segmentation': SegmentationPipeline,
    'detection': DetectionPipeline,
    'pose_estimation': PoseEstimationPipeline,
}
