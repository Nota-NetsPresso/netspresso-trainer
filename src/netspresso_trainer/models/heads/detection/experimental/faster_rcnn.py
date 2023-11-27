from typing import List

from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F

from .detection import AnchorGenerator, RPNHead, RegionProposalNetwork, RoIHeads, GeneralizedRCNN, MultiScaleRoIAlign

IMAGE_SIZE = (512, 512) # TODO: Get from configuration


class FasterRCNN(GeneralizedRCNN):
    def __init__(
        self,
        num_classes: int,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        # Anchor parameters
        anchor_sizes = params.anchor_sizes
        #anchor_sizes = ((64,), (128,), (256,), (512,))
        aspect_ratios = params.aspect_ratios
        #aspect_ratios = (0.5, 1.0, 2.0)
        # RPN parameters
        rpn_pre_nms_top_n = params.rpn_pre_nms_top_n
        rpn_post_nms_top_n = params.rpn_post_nms_top_n
        rpn_nms_thresh = params.rpn_nms_thresh
        rpn_fg_iou_thresh = params.rpn_fg_iou_thresh
        rpn_bg_iou_thresh = params.rpn_bg_iou_thresh
        rpn_batch_size_per_image = params.rpn_batch_size_per_image
        rpn_positive_fraction = params.rpn_positive_fraction
        rpn_score_thresh = params.rpn_score_thresh
        # RoI parameters
        roi_output_size = params.roi_output_size
        roi_sampling_ratio = params.roi_sampling_ratio
        roi_representation_size = params.roi_representation_size
        # Box parameters
        box_score_thresh = params.box_score_thresh
        box_nms_thresh = params.box_nms_thresh
        box_detections_per_img = params.box_detections_per_img
        box_fg_iou_thresh = params.box_fg_iou_thresh
        box_bg_iou_thresh = params.box_bg_iou_thresh
        box_batch_size_per_image = params.box_batch_size_per_image
        box_positive_fraction = params.box_positive_fraction
        bbox_reg_weights = params.bbox_reg_weights

        out_channels = intermediate_features_dim[-1]

        aspect_ratios = (aspect_ratios,) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, IMAGE_SIZE)

        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        featmap_names = [str(i) for i in range(len(intermediate_features_dim))]
        box_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=roi_output_size, sampling_ratio=roi_sampling_ratio)

        box_head = TwoMLPHead(out_channels * roi_output_size**2, roi_representation_size)
        box_predictor = FastRCNNPredictor(roi_representation_size, num_classes)

        roi_heads = RoIHeads(
            num_classes,
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        super().__init__(rpn, roi_heads, IMAGE_SIZE)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def faster_rcnn(num_classes, intermediate_features_dim, conf_model_head, **kwargs) -> FasterRCNN:
    return FasterRCNN(num_classes=num_classes, 
                      intermediate_features_dim=intermediate_features_dim, 
                      params=conf_model_head.params)
