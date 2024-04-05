from typing import Dict, List, Type

from .base import BasePipeline
from .evaluation import EvaluationPipeline
from .inference import InferencePipeline
from .task_processors.base import BaseTaskProcessor
from .task_processors.classification import ClassificationProcessor
from .task_processors.detection import DetectionProcessor
from .task_processors.pose_estimation import PoseEstimationProcessor
from .task_processors.segmentation import SegmentationProcessor
from .train import TrainingPipeline

# TODO: Temporary defined. It should be integrated with `..models.registry.SUPPORTING_TASK_LIST`
SUPPORTING_TASK_LIST: List[str] = ['classification', 'segmentation', 'detection', 'pose_estimation']

TASK_PROCESSOR: Dict[str, Type[BaseTaskProcessor]] = {
    'classification': ClassificationProcessor,
    'segmentation': SegmentationProcessor,
    'detection': DetectionProcessor,
    'pose_estimation': PoseEstimationProcessor,
}

PIPELINES: Dict[str, Type[BasePipeline]] = {
    'train': TrainingPipeline,
    'evaluation': EvaluationPipeline,
    'inference': InferencePipeline,
}
