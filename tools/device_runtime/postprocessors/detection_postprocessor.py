# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from functools import partial

import numpy as np


def anchor_free_decoupled_head_decode(pred, original_shape, score_thresh=0.7):
    pred = pred['pred']
    dtype = pred[0].dtype
    stage_strides= [original_shape[-1] // o.shape[-1] for o in pred]

    hw = [x.shape[-2:] for x in pred]
    dim_len = pred[0].shape[1]

    pred = np.concatenate([x.reshape(1, dim_len, -1) for x in pred], axis=2).transpose(0, 2, 1)
    pred[..., 4:] = 1 / (1 + (np.exp(-pred[..., 4:])))
    
    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, stage_strides):
        yv, xv = np.meshgrid(np.arange(hsize, dtype='float32'), np.arange(wsize, dtype='float32'), indexing='ij')
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(np.full((*shape, 1), stride, dtype='float32'))

    grids = np.concatenate(grids, axis=1)
    strides = np.concatenate(strides, axis=1)

    pred = np.concatenate([
        (pred[..., 0:2] + grids) * strides,
        np.exp(pred[..., 2:4]) * strides,
        pred[..., 4:]
    ], axis=-1)

    box_corner = np.empty(pred.shape)
    box_corner[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
    box_corner[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
    box_corner[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
    box_corner[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
    pred[:, :, :4] = box_corner[:, :, :4]

    # Discard boxes with low score
    detections = []
    for p in pred:
        class_pred = np.argmax(p[:, 5:], 1, keepdims=True)
        class_conf = p[np.arange(p.shape[0]), 5+class_pred.squeeze()][:, np.newaxis]

        conf_mask = (p[:, 4] * class_conf.squeeze() >= score_thresh).squeeze()

        # x1, y1, x2, y2, obj_conf, pred_score, pred_label
        detections.append(
            np.concatenate((p[:, :5], class_conf, class_pred), axis=1)[conf_mask]
        )

    return detections

def nms_fast_rcnn(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms(prediction, nms_thresh=0.45):
    output = [np.zeros((0, 7)) for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue

        nms_out_index = nms_fast_rcnn(
            image_pred[:, :4],
            image_pred[:, 4] * image_pred[:, 5],
            nms_thresh,
        )

        image_pred = image_pred[nms_out_index]
        output[i] = np.concatenate((output[i], image_pred))

    return output


class DetectionPostprocessor:
    def __init__(self, postprocess_conf):
        self.decode_outputs = partial(anchor_free_decoupled_head_decode, score_thresh=postprocess_conf.score_thresh)
        self.postprocess = partial(nms, nms_thresh=postprocess_conf.nms_thresh)

    def __call__(self, outputs, original_shape):
        pred = outputs

        if self.decode_outputs:
            pred = self.decode_outputs(pred, original_shape)
        if self.postprocess:
            pred = self.postprocess(pred)

        pred = [(np.concatenate([p[:, :4], p[:, 4:5] * p[:, 5:6]], axis=-1),
                      p[:, 6].astype('int'))
                      for p in pred]
        return pred
