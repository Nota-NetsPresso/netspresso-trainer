from typing import Dict, Type

from .classification import ClassificationPostprocessor
from .detection import DetectionPostprocessor
from .segmentation import SegmentationPostprocessor

POSTPROCESSOR_DICT = {
    'fc': ClassificationPostprocessor,
    'all_mlp_decoder': SegmentationPostprocessor,
    'anchor_free_decoupled_head': DetectionPostprocessor,
    'pidnet': SegmentationPostprocessor,
    'anchor_decoupled_head': DetectionPostprocessor,
}
