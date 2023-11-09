import torch
import torchvision

from ..models.utils import ModelOutput


def yolox_decode_outputs(pred, original_shape):
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
    return pred


def nms(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    output = [torch.zeros(0, 7).to(prediction.device) for i in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        output[i] = torch.cat((output[i], detections))

    return output


class DetectionPostprocessor:
    def __init__(self, conf_model):
        HEAD_POSTPROCESS_MAPPING = {
            'yolox_head': [yolox_decode_outputs, nms]
        }

        head_name = conf_model.architecture.head.name
        self.decode_outputs, self.postprocess = HEAD_POSTPROCESS_MAPPING[head_name]

    def __call__(self, outputs: ModelOutput, original_shape, num_classes, conf_thresh=0.7, nms_thre=0.45, class_agnostic=False):
        pred = outputs['pred']

        if self.decode_outputs:
            pred = self.decode_outputs(pred, original_shape)
        if self.postprocess:
            pred = self.postprocess(pred, num_classes=num_classes, conf_thre=conf_thresh, nms_thre=nms_thre, class_agnostic=class_agnostic)
        return pred
