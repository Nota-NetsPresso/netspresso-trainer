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


def save_onnx(model: nn.Module, f: Union[str, Path], sample_input: Tensor):
    sample_input = sample_input.to(get_device(model))
    return _save_onnx(model, f, sample_input, opset_version=13, input_names='images', output_names='output')
