import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.fx.proxy import Proxy


def xyxy2cxcywhn(bboxes: Union[Tensor, Proxy], img_size: Union[int, Tuple[int, int]]) -> Union[Tensor, Proxy]:
    if isinstance(img_size, int):
        width = height = img_size
    else:
        width, height = img_size
        assert isinstance(width, int) and isinstance(height, int), f"Invalid type: (width: {type(width)}, height: {type(height)}. Must be (int, int))"


    boxes = bboxes.clone()

    boxes[:, [0, 2]] /= width
    boxes[:, [1, 3]] /= height

    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return torch.stack([cx, cy, w, h], dim=-1)

def xyxy2cxcywh(bboxes: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
    x0, y0, x1, y1 = bboxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def cxcywh2xyxy(bboxes: Union[Tensor, Proxy]) -> Union[Tensor, Proxy]:
    cx, cy, w, h = bboxes.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)

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
    VALID_METRICS = {"iou", "giou", "diou", "ciou"}
    assert metric.lower() in VALID_METRICS, f"Invalid IoU metric: '{metric}'. Must be one of {VALID_METRICS}"
    metric = metric.lower()

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
    elif metric == "diou" or metric == "ciou":
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
        if metric == "diou":
            return diou
        arctan = torch.atan(bbox1[..., 2] / (bbox1[..., 3] + EPS)) - torch.atan(bbox2[..., 2] / (bbox2[..., 3] + EPS))
        v = (4 / (math.pi ** 2)) * (arctan ** 2)
        with torch.no_grad():
            alpha = v / (v - iou + 1 + EPS)
        ciou = diou - alpha * v
        return ciou
