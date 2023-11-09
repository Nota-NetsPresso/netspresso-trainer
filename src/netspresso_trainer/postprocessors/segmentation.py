from typing import Any, Optional

import torch

from ..models.utils import ModelOutput


class SegmentationPostprocessor:
    def __init__(self, conf_model):
        pass

    def __call__(self, outputs: ModelOutput):
        pred = outputs['pred']
        pred = torch.max(pred, dim=1)[1]  # argmax
        return pred
