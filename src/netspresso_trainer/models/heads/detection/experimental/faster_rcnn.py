import torch.nn as nn
import torch.nn.functional as F

from .detection import AnchorGenerator, RPNHead, RegionProposalNetwork, RoIHeads, GeneralizedRCNN, MultiScaleRoIAlign
from .fpn import FPN

IMAGE_SIZE = (512, 512) # TODO: Get from configuration


class FasterRCNN(GeneralizedRCNN):
    def __init__(
        self,
        num_classes,
        intermediate_features_dim,
        # FPN parameters
        fpn_num_outs=4,
        # Anchor parameters
        anchor_sizes=((64,), (128,), (256,), (512,)),
        aspect_ratios=(0.5, 1.0, 2.0),
        # RPN parameters
        rpn_pre_nms_top_n=2000,
        rpn_post_nms_top_n=2000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # RoI parameters
        roi_output_size=7,
        roi_sampling_ratio=2,
        roi_representation_size=1024,
        # Box parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):
        assert fpn_num_outs == len(anchor_sizes)

        neck = FPN(in_channels=intermediate_features_dim, out_channels=intermediate_features_dim[-1], num_outs=fpn_num_outs)

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

        featmap_names = [str(i) for i in range(neck.num_outs)]
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

        super().__init__(neck, rpn, roi_heads, IMAGE_SIZE)


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


def faster_rcnn(num_classes, intermediate_features_dim, **kwargs):
    configuration = {
        # FPN parameters
        'fpn_num_outs': 4,
        # Anchor parameters
        'anchor_sizes': ((64,), (128,), (256,), (512,)),
        'aspect_ratios': (0.5, 1.0, 2.0),
        # RPN parameters
        'rpn_pre_nms_top_n': 2000,
        'rpn_post_nms_top_n': 2000,
        'rpn_nms_thresh': 0.7,
        'rpn_fg_iou_thresh': 0.7,
        'rpn_bg_iou_thresh': 0.3,
        'rpn_batch_size_per_image': 256,
        'rpn_positive_fraction': 0.5,
        'rpn_score_thresh': 0.0,
        # RoI parameters
        'roi_output_size': 7,
        'roi_sampling_ratio': 2,
        'roi_representation_size': 1024,
        # Box parameters
        'box_score_thresh': 0.05,
        'box_nms_thresh': 0.5,
        'box_detections_per_img': 100,
        'box_fg_iou_thresh': 0.5,
        'box_bg_iou_thresh': 0.5,
        'box_batch_size_per_image': 512,
        'box_positive_fraction': 0.25,
        'bbox_reg_weights': None,
    }
    return FasterRCNN(num_classes=num_classes, intermediate_features_dim=intermediate_features_dim, **configuration)
