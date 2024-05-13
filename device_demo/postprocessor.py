from functools import partial

import torch
import torchvision


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
        torch.exp(pred[..., 2:4]) * strides,
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
    def __init__(self, score_thresh, nms_thresh, class_agnostic):
        self.decode_outputs = partial(anchor_free_decoupled_head_decode, score_thresh=score_thresh)
        self.postprocess = partial(nms, nms_thresh=nms_thresh, class_agnostic=class_agnostic)

    def __call__(self, outputs, original_shape):
        pred = outputs

        if self.decode_outputs:
            pred = self.decode_outputs(pred, original_shape)
        if self.postprocess:
            pred = self.postprocess(pred)

        pred = [(torch.cat([p[:, :4], p[:, 4:5] * p[:, 5:6]], dim=-1).detach().cpu().numpy(),
                      p[:, 6].to(torch.int).detach().cpu().numpy())
                      for p in pred]
        return pred
