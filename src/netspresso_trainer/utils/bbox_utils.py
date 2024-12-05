import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.fx.proxy import Proxy


def generate_anchors(image_shape: Union[int, Tuple[int, int]], strides: List[int]):
    """
        This implementation is based on https://github.com/WongKinYiu/YOLO/blob/main/yolo/utils/bounding_box_utils.py.
    """
    image_height, image_width = image_shape
    anchors = []
    scaler = []

    for stride in strides:
        anchor_num = image_width // stride * image_height // stride
        scaler.append(torch.full((anchor_num,), stride))
        offset = stride // 2
        h = torch.arange(0, image_height, stride) + offset
        w = torch.arange(0, image_width, stride) + offset

        if torch.__version__ >= "2.3.0":
            anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        else:
            anchor_h, anchor_w = torch.meshgrid(h, w)

        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers

def transform_bbox(bboxes: Union[Tensor, Proxy],
                   indicator="xywh -> xyxy",
                   image_shape: Optional[Union[int, Tuple[int, int]]]=None):
    def is_normalized(fmt: str) -> bool:
        return fmt.endswith('n')

    VALID_IN_TYPE = VALID_OUT_TYPE = ["xyxy", "xyxyn", "xywh", "xywhn", "cxcywh", "cxcywhn"]
    dtype = bboxes.dtype
    in_type, out_type = indicator.replace(" ", "").split("->")
    assert in_type in VALID_IN_TYPE, f"Invalid in_type: '{in_type}'. Must be one of {VALID_IN_TYPE}."
    assert out_type in VALID_OUT_TYPE, f"Invalid out_type: '{out_type}'. Must be one of {VALID_OUT_TYPE}."

    if is_normalized(in_type):
        assert image_shape is not None, f"image_shape is required for normalized conversion: {indicator}"
        if isinstance(image_shape, int):
            img_height = img_width = image_shape
        else:
            img_height, img_width = image_shape
            assert isinstance(img_height, int) and isinstance(img_width, int), \
                f"Invalid type: (height: {type(img_height)}, width: {type(img_width)}. Must be (int, int))"
        in_type = in_type[:-1]
    else:
        img_height = img_width = 1.0

    if in_type == "xyxy":
        x_min, y_min, x_max, y_max = bboxes.unbind(-1)
    elif in_type == "xywh":
        x_min, y_min, w, h = bboxes.unbind(-1)
        x_max = x_min + w
        y_max = y_min + h
    elif in_type == "cxcywh":
        cx, cy, w, h = bboxes.unbind(-1)
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

    x_min *= img_width
    y_min *= img_height
    x_max *= img_width
    y_max *= img_height
    assert (x_max >= x_min).all(), "Invalid box: x_max < x_min"
    assert (y_max >= y_min).all(), "Invalid box: y_max < y_min"

    if is_normalized(out_type):
        assert image_shape is not None, f"img_size is required for normalized conversion: {indicator}"
        if isinstance(image_shape, int):
            img_height = img_width = image_shape
        else:
            img_height, img_width = image_shape
            assert isinstance(img_height, int) and isinstance(img_width, int), \
                f"Invalid type: (height: {type(img_height)}, width: {type(img_width)}. Must be (int, int))"
        out_type = out_type[:-1]
    else:
        img_height = img_width = 1.0

    x_min /= img_width
    y_min /= img_height
    x_max /= img_width
    y_max /= img_height
    if out_type == "xywh":
        bbox = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=-1)
    elif out_type == "xyxy":
        bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    elif out_type == "cxcywh":
        bbox = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=-1)

    return bbox.to(dtype=dtype)

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
