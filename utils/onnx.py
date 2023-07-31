import torch

def get_device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, torch.nn.Module):
        return next(x.parameters()).device
    raise RuntimeError(f'{type(x)} do not have `device`')

def _save_onnx(model, f, sample_input):
    torch.onnx.export(model,  # model being run
                      sample_input,  # model input (or a tuple for multiple inputs)
                      f,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['images'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'images': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

def save_onnx(model, f, sample_input):
    sample_input = sample_input.to(get_device(model))
    return _save_onnx(model, f, sample_input)
