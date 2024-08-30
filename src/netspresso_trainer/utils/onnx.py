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
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from .environment import get_device

__all__ = ['save_onnx']


def _save_onnx(model: nn.Module, f: Union[str, Path], sample_input: Tensor,
               opset_version=13, input_names='images', output_names='output'):
    torch.onnx.export(model,  # model being run
                      sample_input,  # model input (or a tuple for multiple inputs)
                      f,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=opset_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=[input_names],  # the model's input names
                      output_names=[output_names],  # the model's output names
                      dynamic_axes={input_names: {0: 'batch_size'},  # variable length axes
                                    output_names: {0: 'batch_size'}})


def save_onnx(model: nn.Module, f: Union[str, Path], sample_input: Tensor, opset_version):
    sample_input = sample_input.to(get_device(model))
    return _save_onnx(model, f, sample_input, opset_version=opset_version, input_names='images', output_names='output')
