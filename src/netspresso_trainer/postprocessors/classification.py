from typing import Optional

from ..models.utils import ModelOutput

TOPK_MAX = 20


class ClassificationPostprocessor():
    def __init__(self, conf_model):
        pass

    def __call__(self, outputs: ModelOutput, k: Optional[int]=None):
        pred = outputs['pred']
        maxk = min(TOPK_MAX, pred.size()[1])
        if k:
            maxk = min(k, maxk)
        _, pred = pred.topk(maxk, 1, True, True)
        return pred
