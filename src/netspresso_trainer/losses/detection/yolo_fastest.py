import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Dict
from .yolox import bboxes_iou, xyxy2cxcywh


def iou_width_height(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """ 
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = (boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection)
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

class YOLOFastestLoss(nn.Module):
    def __init__(self, cur_epoch=None, anchors=None, **kwargs) -> None:
        super(YOLOFastestLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.anchors = anchors
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.ignore_iou_threshold = 0.5

    def forward(self, preds: List, target: Dict) -> torch.Tensor:
        total_loss = torch.zeros(1, device=preds[0].device)
        img_size = target['img_size']
        gt = target['gt']
        self.grids = [torch.zeros(1)] * len(preds) # 각 layer 별로 grid를 생성함.
        self.num_classes = target['num_classes']
        x_shifts, y_shifts, expanded_strides, out_for_loss = self.get_offsets_and_out(preds, img_size)

        num_channels = [o.shape[-1] for o in preds]
        objs1 = []
        objs2 = []
        for idx in range(len(gt)):
            gt[idx]['boxes'] = xyxy2cxcywh(gt[idx]['boxes'])
            obj1, obj2 = self.get_objectness_indicators(gt[idx]['boxes'], gt[idx]['labels'], self.anchors, num_channels, img_size)
            objs1.append(obj1)
            objs2.append(obj2)
        objs1 = torch.stack(objs1, dim=0)
        objs2 = torch.stack(objs2, dim=0)
        objs = [objs1, objs2]

        for idx, obj in enumerate(objs):
           total_loss += self.get_losses(out_for_loss[idx], obj, num_channels[idx])
        
        return total_loss

    def get_objectness_indicators(self, bboxes, labels, anchors, num_channels, img_size):
        targets = [torch.zeros((self.num_anchors, nc, nc, 6)) for nc in num_channels]
        anchors = [[tuple(anchor[i : i + 2]) for i in range(0, len(anchor), 2)] for anchor in anchors]
        anchors = torch.tensor(anchors[0] + anchors[1])
        for box, class_label in zip(bboxes, labels):
            iou_anchors = iou_width_height(box[2:4], anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height = box
            has_anchor = [False] * 2

        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // self.num_anchors
            anchor_on_scale = anchor_idx % self.num_anchors
            S = num_channels[scale_idx]
            i, j = int(S * y/img_size), int(S * x/img_size) # which cell

            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if not anchor_taken and not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S * x - j, S * y - i # both between [0,1]
                width_cell, height_cell = (width, height,) # can be greater than 1 since it's relative to cell
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                has_anchor[scale_idx] = True
            elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 0 # ignore prediction

        return targets


    def get_losses(self, preds, objs, num_channel):
        pred = preds.view(-1, self.num_anchors, num_channel, num_channel, 5+self.num_classes)
        target = objs.view(-1, self.num_anchors, num_channel, num_channel, 6)
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        no_object_loss = self.bce(
            pred[..., 4:5][noobj], target[..., 0:1][noobj],
        )

        box_preds = pred[..., :4]
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()

        object_loss = self.mse(self.sigmoid(pred[..., 4:5][obj]), ious * target[..., 0:1][obj])
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])
        class_loss = self.entropy(
            pred[..., 5:][obj], target[..., 5][obj].long()
        )

        return self.lambda_noobj * no_object_loss + self.lambda_coord * box_loss + class_loss + object_loss


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        num_anchors = self.num_anchors
        anchors = torch.tensor(self.anchors[k]).to(output.device)
        anchors = anchors.view(1, num_anchors, 1, 2)
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        output = output.view(batch_size, self.num_anchors, n_ch, hsize, wsize) # [8, 3, 85, h, w]
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, hsize * wsize, n_ch)
        grid = grid.view(1, 1, -1, 2)
        output = torch.cat([
                            torch.sigmoid(output[..., :2]) + grid, # cx and cy
                            torch.exp(output[..., 2:4]) * anchors, # box_width and box_height
                            output[..., 4:5], # confidence score: sigmoid를 사용하지 않는 이유는 BCE with logit loss를 사용하기 때문이다.
                            output[..., 5:] # cls prob
                            ], dim=-1)
        return output, grid


    def get_offsets_and_out(self, preds, img_size):
        x_shifts = list()
        y_shifts = list()
        out_for_loss = list()
        expanded_strides = list()
        for k, o in enumerate(preds):
            stride_this_level = img_size // o.size(-1)
            o, grid = self.get_output_and_grid(o, k, stride_this_level, o.type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                            torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(o))
            out_for_loss.append(o)
        # out_for_loss: [2, B, na, c*c, 85]

        return x_shifts, y_shifts, expanded_strides, out_for_loss