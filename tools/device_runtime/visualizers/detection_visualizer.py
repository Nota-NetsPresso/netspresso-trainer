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

import cv2
import numpy as np


def _voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class DetectionVisualizer:
    def __init__(self, class_map=None):
        self.cmap = _voc_color_map(256)
        self.class_map = class_map

    def draw(self, image, detections, model_input_shape):
        resize_factor = max((image.shape[0] / model_input_shape[0]), (image.shape[1] / model_input_shape[1]))
        detections[0][:, 1::2] *= resize_factor
        detections[0][:, :4:2] *= resize_factor

        visualize_image = image.copy()
        for bbox_label, class_label in zip(detections[0], detections[1]):
            # Get bbox coordinates
            x1 = int(bbox_label[0])
            y1 = int(bbox_label[1])
            x2 = int(bbox_label[2])
            y2 = int(bbox_label[3])

            # Get bbox color
            color = self.cmap[class_label].tolist()

            # Draw bbox
            visualize_image = cv2.rectangle(visualize_image, (x1, y1), (x2, y2), color=color, thickness=2)

            # Get class name
            class_name = self.class_map[class_label] if self.class_map else str(class_label)

            # Draw class info
            text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            visualize_image = cv2.rectangle(visualize_image, (x1, y1-5-text_h), (x1+text_w, y1), color=color, thickness=-1)
            visualize_image = cv2.putText(visualize_image, str(class_name), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return visualize_image

    def visualize(self, image):
        cv2.imshow('Detecttion result', image)
        cv2.waitKey(1)