"""
Implements the Generalized R-CNN framework
"""
from typing import Tuple

from torch import nn

from .....utils import DetectionModelOutput, FXTensorListType


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
    """

    def __init__(self, neck:nn.Module, rpn: nn.Module, roi_heads: nn.Module, image_size: Tuple[int, int]) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.neck = neck
        self.rpn = rpn
        self.roi_heads = roi_heads

        self.image_size = image_size

    def forward(self, features: FXTensorListType) -> DetectionModelOutput:
        if self.neck:
            features = self.neck(features)

        features = {str(k): v for k, v in enumerate(features)}
        rpn_features = self.rpn(features, self.image_size)
        roi_features = self.roi_heads(features, rpn_features['boxes'], self.image_size)

        out_features = DetectionModelOutput()
        out_features.update(rpn_features)
        out_features.update(roi_features)

        return out_features