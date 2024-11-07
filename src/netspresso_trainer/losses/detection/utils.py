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

def xyxy2cxcywh(bboxes: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)