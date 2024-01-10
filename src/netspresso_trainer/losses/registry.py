from .common import CrossEntropyLoss, SigmoidFocalLoss
from .detection import RetinaNetLoss, YOLOXLoss
from .segmentation import PIDNetLoss

LOSS_DICT = {
    'cross_entropy': CrossEntropyLoss,
    'pidnet_loss': PIDNetLoss,
    'yolox_loss': YOLOXLoss,
    'retinanet_loss': RetinaNetLoss,
    'focal_loss': SigmoidFocalLoss,
}

PHASE_LIST = ['train', 'valid', 'test']
