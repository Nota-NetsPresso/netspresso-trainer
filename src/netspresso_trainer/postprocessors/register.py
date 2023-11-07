from typing import Dict, Type

from .classification import ClassificationPostprocessor
from .detection import YOLOXPostprocessor
from .segmentation import SegmentationPostprocessor

POSTPROCESSOR_DICT = {
    'fc': ClassificationPostprocessor,
    'all_mlp_decoder': SegmentationPostprocessor,
    'yolo_head': YOLOXPostprocessor,
}
