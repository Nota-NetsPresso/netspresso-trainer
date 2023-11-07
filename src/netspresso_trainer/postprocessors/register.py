from typing import Dict, Type

from .classification import ClassificationPostprocessor
from .segmentation import SegmentationPostprocessor
from .detection import YOLOXPostprocessor

POSTPROCESSOR_DICT = {
    'fc': ClassificationPostprocessor,
    'all_mlp_decoder': SegmentationPostprocessor,
    'yolo_head': YOLOXPostprocessor,
}