from typing import Dict, Type

from .classification import ClassificationPostprocessor
from .detection import DetectionPostprocessor
from .segmentation import SegmentationPostprocessor

POSTPROCESSOR_DICT = {
    'fc': ClassificationPostprocessor,
    'all_mlp_decoder': SegmentationPostprocessor,
    'yolox_head': DetectionPostprocessor,
    'pidnet': SegmentationPostprocessor,
}
