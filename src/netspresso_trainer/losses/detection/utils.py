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


def calculate_iou(bbox1, bbox2, metric="iou", EPS=1e-7) -> Tensor:
    VALID_METRICS = {"iou", "giou", "diou"}
    assert metric in VALID_METRICS, f"Invalid IoU metric: '{metric}'. Must be one of {VALID_METRICS}"
    tl = torch.max(
        (bbox1[:, :2] - bbox1[:, 2:] / 2), (bbox2[:, :2] - bbox2[:, 2:] / 2)
    )
    br = torch.min(
        (bbox1[:, :2] + bbox1[:, 2:] / 2), (bbox2[:, :2] + bbox2[:, 2:] / 2)
    )

    area_p = torch.prod(bbox1[:, 2:], 1)
    area_g = torch.prod(bbox2[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    area_u = area_p + area_g - area_i
    iou = (area_i) / (area_u + EPS)

    if metric == "iou":
        return iou
    elif metric == "giou":
        c_tl = torch.min(
            (bbox1[:, :2] - bbox1[:, 2:] / 2), (bbox2[:, :2] - bbox2[:, 2:] / 2)
        )
        c_br = torch.max(
            (bbox1[:, :2] + bbox1[:, 2:] / 2), (bbox2[:, :2] + bbox2[:, 2:] / 2)
        )
        area_c = torch.prod(c_br - c_tl, 1)
        giou = iou - (area_c - area_u) / area_c.clamp(EPS)
        return giou
    elif metric == "diou":
        cent1 = bbox1[..., :2]  # (cx1, cy1)
        cent2 = bbox2[..., :2]  # (cx2, cy2)

        cent_dist = torch.sum((cent1 - cent2) * (cent1 - cent2), dim=-1)

        c_tl = torch.min(
            bbox1[..., :2] - bbox1[..., 2:] / 2,
            bbox2[..., :2] - bbox2[..., 2:] / 2
        )
        c_br = torch.max(
            bbox1[..., :2] + bbox1[..., 2:] / 2,
            bbox2[..., :2] + bbox2[..., 2:] / 2
        )

        diag_dist = torch.sum((c_br - c_tl) ** 2, dim=-1) + EPS

        diou = iou - (cent_dist / diag_dist)
        return diou
