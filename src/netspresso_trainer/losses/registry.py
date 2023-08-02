from losses.common import CrossEntropyLoss
from losses.classification.label_smooth import LabelSmoothingCrossEntropy
from losses.classification.soft_target import SoftTargetCrossEntropy
from losses.segmentation.pidnet import PIDNetCrossEntropy, PIDNetBoundaryAwareCrossEntropy, BondaryLoss
from losses.detection.fastrcnn import RoiHeadLoss, RPNLoss

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'soft_target_cross_entropy': SoftTargetCrossEntropy,
    'label_smoothing_cross_entropy': LabelSmoothingCrossEntropy,
    'pidnet_cross_entropy': PIDNetCrossEntropy,
    'boundary_loss': BondaryLoss,
    'pidnet_cross_entropy_with_boundary': PIDNetBoundaryAwareCrossEntropy,
    'roi_head_loss': RoiHeadLoss,
    'rpn_loss': RPNLoss,
}