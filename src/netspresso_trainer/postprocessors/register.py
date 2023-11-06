from typing import Dict, Type

from .classification import TopK

POSTPROCESSOR_DICT = {
    'fc': TopK,
}