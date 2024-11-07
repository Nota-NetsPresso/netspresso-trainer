from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.fx.proxy import Proxy


def xyxy2cxcywhn(bboxes: Union[Tensor, Proxy], img_size: Union[int, Tuple[int, int]]) -> Union[Tensor, Proxy]:
    if isinstance(img_size, int):
        img_size = (img_size, img_size) # [w, h]
    assert isinstance(img_size, Tuple)

    new_bboxes = bboxes.clone()
    new_bboxes[:, ::2] /= img_size[0]
    new_bboxes[:, 1::2] /= img_size[1]
    new_bboxes[:, 2] = new_bboxes[:, 2] - new_bboxes[:, 0]
    new_bboxes[:, 3] = new_bboxes[:, 3] - new_bboxes[:, 1]
    new_bboxes[:, 0] = new_bboxes[:, 0] + new_bboxes[:, 2] * 0.5
    new_bboxes[:, 1] = new_bboxes[:, 1] + new_bboxes[:, 3] * 0.5

    return new_bboxes
