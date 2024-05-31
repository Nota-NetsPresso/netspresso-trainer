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

import torch
from torch.nn import functional as F


def classification_mix_collate_fn(original_batch, mix_transforms):
    indices = []
    images = []
    target = []
    for data_sample in original_batch:
        indices.append(data_sample[0])
        images.append(data_sample[1])
        target.append(data_sample[2])

    indices = torch.tensor(indices, dtype=torch.long)
    images = torch.stack(images, dim=0)
    target = torch.tensor(target, dtype=torch.long)

    images, target = mix_transforms(images, target)

    outputs = (indices, images, target)
    return outputs


def classification_onehot_collate_fn(original_batch, num_classes):
    indices = []
    images = []
    target = []
    for data_sample in original_batch:
        indices.append(data_sample[0])
        images.append(data_sample[1])
        target.append(data_sample[2])

    indices = torch.tensor(indices, dtype=torch.long)
    images = torch.stack(images, dim=0)
    target = torch.tensor(target, dtype=torch.long)
    if -1 not in target:
        target = F.one_hot(target, num_classes=num_classes).to(dtype=images.dtype)

    outputs = (indices, images, target)
    return outputs


def detection_collate_fn(original_batch):
    indices = []
    pixel_values = []
    bbox = []
    label = []
    org_shape = []
    for data_sample in original_batch:
        if 'indices' in data_sample:
            indices.append(data_sample['indices'])
        if 'pixel_values' in data_sample:
            pixel_values.append(data_sample['pixel_values'])
        if 'bbox' in data_sample:
            bbox.append(data_sample['bbox'])
        if 'label' in data_sample:
            label.append(data_sample['label'])
        if 'org_shape' in data_sample:
            org_shape.append(data_sample['org_shape'])
    outputs = {}
    if len(indices) != 0:
        indices = torch.tensor(indices, dtype=torch.long)
        outputs.update({'indices': indices})
    if len(pixel_values) != 0:
        pixel_values = torch.stack(pixel_values, dim=0)
        outputs.update({'pixel_values': pixel_values})
    if len(bbox) != 0:
        outputs.update({'bbox': bbox})
    if len(label) != 0:
        outputs.update({'label': label})
    if len(org_shape) != 0:
        outputs.update({'org_shape': org_shape})

    return outputs
