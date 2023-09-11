from .common import CrossEntropyLoss
from .classification import LabelSmoothingCrossEntropy
from .classification import SoftTargetCrossEntropy
from .segmentation import PIDNetCrossEntropy, PIDNetBoundaryAwareCrossEntropy, BondaryLoss
from .detection import RoiHeadLoss, RPNLoss

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

PHASE_LIST = ['train', 'valid', 'test']