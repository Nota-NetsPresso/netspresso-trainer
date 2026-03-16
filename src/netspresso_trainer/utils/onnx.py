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

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

from .environment import get_device

__all__ = ['save_onnx']


def _save_onnx(model: nn.Module, f: Union[str, Path], sample_input: Tensor,
               opset_version=13, input_names='images',
               output_names: Union[str, List[str]] = 'output'):
    if isinstance(output_names, str):
        output_names = [output_names]
    dynamic_axes = {input_names: {0: 'batch_size'}}
    dynamic_axes.update({name: {0: 'batch_size'} for name in output_names})
    torch.onnx.export(model,
                      sample_input,
                      f,
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=[input_names],
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)


def save_onnx(model: nn.Module, f: Union[str, Path], sample_input: Tensor, opset_version):
    sample_input = sample_input.to(get_device(model))
    # When the head is in export mode it returns (boxes, class_scores).
    head = getattr(model, 'head', None)
    if head is not None and getattr(head, '_export', False):
        output_names = ['output', 'class_scores']
    else:
        output_names = ['output']
    return _save_onnx(model, f, sample_input, opset_version=opset_version,
                      input_names='images', output_names=output_names)
