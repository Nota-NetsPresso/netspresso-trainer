from .common import CrossEntropyLoss, SigmoidFocalLoss
from .detection import RetinaNetLoss, YOLOXLoss
from .segmentation import BoundaryLoss, PIDNetBoundaryAwareCrossEntropy, PIDNetCrossEntropy

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'pidnet_cross_entropy': PIDNetCrossEntropy,
    'boundary_loss': BoundaryLoss,
    'pidnet_cross_entropy_with_boundary': PIDNetBoundaryAwareCrossEntropy,
    'yolox_loss': YOLOXLoss,
    'retinanet_loss': RetinaNetLoss,
    'focal_loss': SigmoidFocalLoss,
}

PHASE_LIST = ['train', 'valid', 'test']
