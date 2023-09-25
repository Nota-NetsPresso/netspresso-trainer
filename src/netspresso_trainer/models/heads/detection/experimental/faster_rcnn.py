import copy
import warnings
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import MultiScaleRoIAlign

from ....utils import DetectionModelOutput, FXTensorListType
from .detection import FasterRCNN, MaskRCNNHeads, MaskRCNNPredictor
from .fpn import FPN


class DetectionHead(FasterRCNN):
    def __init__(
        self,
        num_classes,
        intermediate_features_dim,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        **kwargs,
    ):

        if not isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(mask_roi_pool)}"
            )

        super().__init__(
            num_classes,
            intermediate_features_dim[-1],
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            **kwargs,
        )

        self.neck = FPN(in_channels=intermediate_features_dim, out_channels=intermediate_features_dim[-1], num_outs=4)

    def forward(self, features: FXTensorListType, targets) -> DetectionModelOutput:
        assert targets is not None
        features = self.neck(features)
        features = {str(k): v for k, v in enumerate(features)}
        rpn_features = self.rpn(features)
        roi_features = self.roi_heads(features, rpn_features['proposals'], [self.image_size] * features["0"].size(0), targets=targets)

        out_features = DetectionModelOutput()
        out_features.update(rpn_features)
        out_features.update(roi_features)

        return out_features


def faster_rcnn(num_classes, intermediate_features_dim, **kwargs):
    return DetectionHead(num_classes=num_classes, intermediate_features_dim=intermediate_features_dim)
