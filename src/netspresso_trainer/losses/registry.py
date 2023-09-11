from .classification import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .common import CrossEntropyLoss
from .detection import RoiHeadLoss, RPNLoss
from .segmentation import BoundaryLoss, PIDNetBoundaryAwareCrossEntropy, PIDNetCrossEntropy

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'soft_target_cross_entropy': SoftTargetCrossEntropy,
    'label_smoothing_cross_entropy': LabelSmoothingCrossEntropy,
    'pidnet_cross_entropy': PIDNetCrossEntropy,
    'boundary_loss': BoundaryLoss,
    'pidnet_cross_entropy_with_boundary': PIDNetBoundaryAwareCrossEntropy,
    'roi_head_loss': RoiHeadLoss,
    'rpn_loss': RPNLoss,
}

PHASE_LIST = ['train', 'valid', 'test']