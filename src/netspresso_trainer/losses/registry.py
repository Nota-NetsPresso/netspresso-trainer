from .common import CrossEntropyLoss, SigmoidFocalLoss
from .detection import RoiHeadLoss, RPNLoss, YOLOXLoss, RetinaNetLoss
from .segmentation import BoundaryLoss, PIDNetBoundaryAwareCrossEntropy, PIDNetCrossEntropy

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'pidnet_cross_entropy': PIDNetCrossEntropy,
    'boundary_loss': BoundaryLoss,
    'pidnet_cross_entropy_with_boundary': PIDNetBoundaryAwareCrossEntropy,
    'roi_head_loss': RoiHeadLoss,
    'rpn_loss': RPNLoss,
    'yolox_loss': YOLOXLoss,
    'retinanet_loss': RetinaNetLoss,
    'focal_loss': SigmoidFocalLoss,
}

PHASE_LIST = ['train', 'valid', 'test']
