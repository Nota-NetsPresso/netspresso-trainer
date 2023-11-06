from typing import Dict, Type

from .classification import TopK
from .segmentation import SegmentationArgMax

POSTPROCESSOR_DICT = {
    'fc': TopK,
    'all_mlp_decoder': SegmentationArgMax
}