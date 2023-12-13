from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection._utils import BoxCoder, _topk_min
from torchvision.ops import boxes as box_ops

from ..models.utils import ModelOutput


def anchor_decoupled_head_decode(pred, original_shape, topk_candidates=1000, score_thresh=0.05):
    box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    class_logits = pred["cls_logits"]
    box_regression = pred["bbox_regression"]
    anchors = pred["anchors"]

    detections = []

    for index in range(len(box_regression[0])):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image = anchors.split([b.shape[0] for b in box_regression_per_image])

        image_boxes = []
        image_scores = []
        image_labels = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = _topk_min(topk_idxs, topk_candidates, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, original_shape[1:])

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0).to(image_scores.dtype)
        tmp_obj = torch.ones(image_labels.shape, dtype=image_labels.dtype, device=image_labels.device).view(-1, 1)

        # x1, y1, x2, y2, dummy, pred_score, pred_label
        detections.append(
            torch.cat([image_boxes, tmp_obj.view(-1, 1), image_scores.view(-1, 1), image_labels.view(-1, 1)], dim=1)
        )

    return detections


def anchor_free_decoupled_head_decode(pred, original_shape, score_thresh=0.7):
    pred = pred['pred']
    dtype = pred[0].type()
    stage_strides= [original_shape[-1] // o.shape[-1] for o in pred]

    hw = [x.shape[-2:] for x in pred]
    # [batch, n_anchors_all, num_classes + 5]
    pred = torch.cat([x.flatten(start_dim=2) for x in pred], dim=2).permute(0, 2, 1)
    pred[..., 4:] = pred[..., 4:].sigmoid()

    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, stage_strides):
        yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    pred = torch.cat([
        (pred[..., 0:2] + grids) * strides,
        torch.clamp(torch.exp(pred[..., 2:4]) * strides, min=torch.iinfo(torch.int32).min, max=torch.iinfo(torch.int32).max),
        pred[..., 4:]
    ], dim=-1)

    box_corner = pred.new(pred.shape)
    box_corner[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
    box_corner[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
    box_corner[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
    box_corner[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
    pred[:, :, :4] = box_corner[:, :, :4]

    # Discard boxes with low score
    detections = []
    for p in pred:
        class_conf, class_pred = torch.max(p[:, 5:], 1, keepdim=True)

        conf_mask = (p[:, 4] * class_conf.squeeze() >= score_thresh).squeeze()

        # x1, y1, x2, y2, obj_conf, pred_score, pred_label
        detections.append(
            torch.cat((p[:, :5], class_conf, class_pred.float()), 1)[conf_mask]
        )

    return detections


def nms(prediction, nms_thresh=0.45, class_agnostic=False):
    output = [torch.zeros(0, 7).to(prediction[0].device) for i in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                image_pred[:, :4],
                image_pred[:, 4] * image_pred[:, 5],
                nms_thresh,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                image_pred[:, :4],
                image_pred[:, 4] * image_pred[:, 5],
                image_pred[:, 6],
                nms_thresh,
            )

        image_pred = image_pred[nms_out_index]
        output[i] = torch.cat((output[i], image_pred))

    return output


class DetectionPostprocessor:
    def __init__(self, conf_model):
        head_name = conf_model.architecture.head.name
        params = conf_model.architecture.head.params
        if head_name == 'anchor_free_decoupled_head':
            self.decode_outputs = partial(anchor_free_decoupled_head_decode, score_thresh=params.score_thresh)
            self.postprocess = partial(nms, nms_thresh=params.nms_thresh, class_agnostic=params.class_agnostic)
        elif head_name == 'anchor_decoupled_head':
            self.decode_outputs = partial(anchor_decoupled_head_decode, topk_candidates=params.topk_candidates, score_thresh=params.score_thresh)
            self.postprocess = partial(nms, nms_thresh=params.nms_thresh, class_agnostic=params.class_agnostic)
        else:
            self.decode_outputs = None
            self.postprocess = None

    def __call__(self, outputs: ModelOutput, original_shape):
        pred = outputs

        if self.decode_outputs:
            pred = self.decode_outputs(pred, original_shape)
        if self.postprocess:
            pred = self.postprocess(pred)

        pred = [(torch.cat([p[:, :4], p[:, 5:6]], dim=-1).detach().cpu().numpy(),
                      p[:, 6].to(torch.int).detach().cpu().numpy())
                      for p in pred]
        return pred
