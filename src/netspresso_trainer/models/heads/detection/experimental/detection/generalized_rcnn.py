"""
Implements the Generalized R-CNN framework
"""
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor, nn

from .....utils import DetectionModelOutput, FXTensorListType

IMAGE_SIZE = (512, 512)

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
    """

    def __init__(self, neck:nn.Module, rpn: nn.Module, roi_heads: nn.Module) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.neck = neck
        self.rpn = rpn
        self.roi_heads = roi_heads

        self.image_size = IMAGE_SIZE  # TODO: get from configuration

    def forward(self, features: FXTensorListType, targets) -> DetectionModelOutput:
        assert targets is not None
        if self.neck:
            features = self.neck(features)

        features = {str(k): v for k, v in enumerate(features)}
        rpn_features = self.rpn(features)
        roi_features = self.roi_heads(features, rpn_features['proposals'], [self.image_size] * features["0"].size(0), targets=targets)

        out_features = DetectionModelOutput()
        out_features.update(rpn_features)
        out_features.update(roi_features)

        return out_features