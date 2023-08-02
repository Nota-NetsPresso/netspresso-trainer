from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


__all__ = ['get_device', 'save_onnx']

def get_device(x: Union[Tensor, nn.Module]):
    if isinstance(x, Tensor):
        return x.device
    if isinstance(x, nn.Module):
        return next(x.parameters()).device
    raise RuntimeError(f'{type(x)} do not have `device`')

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
