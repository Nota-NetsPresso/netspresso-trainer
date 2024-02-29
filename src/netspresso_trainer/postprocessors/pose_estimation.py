from typing import Optional

from ..models.utils import ModelOutput

TOPK_MAX = 20


class PoseEstimationPostprocessor():
    def __init__(self, conf_model):
        pass

    def __call__(self, outputs: ModelOutput, k: Optional[int]=None):
        pred = outputs['pred']
        return pred
