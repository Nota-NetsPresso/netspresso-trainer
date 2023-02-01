from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.fx as fx
from omegaconf import OmegaConf

def convert_onnx(args, model, batch, channel, width, height):
    _save_path = Path(args.convert.save_path.weight.dir) / args.convert.save_path.weight.onnx

    dummy_input = torch.randn(batch, channel, width, height)
    try:
        torch.onnx.export(model, dummy_input, _save_path, verbose=args.convert.verbose, input_names=['input'], output_names=['output'])
    except Exception as e:
        raise e
    
def _save_graphmodule(model, save_path):
    torch.save(model, save_path)
    
def save_graphmodule(model, save_path):
    return _save_graphmodule(model, save_path)
    
def _convert_graphmodule(model):
    
    try:
        _graph = fx.Tracer().trace(model)
        model = fx.GraphModule(model, _graph)
        return model
    except Exception as e:
        raise e
    
def convert_graphmodule(model, save_path, save_module=True):
    graphmodule_model = _convert_graphmodule(model)
    if save_module:
        _save_graphmodule(graphmodule_model, save_path)

def convert_graphmodule_with_args(args, model, save_module=True):

    _save_path = Path(args.convert.save_path.weight.dir) / args.convert.save_path.weight.graphmodule

    graphmodule_model = _convert_graphmodule(model)
    if save_module:
        _save_graphmodule(graphmodule_model, _save_path)

def save_model_info_for_nptk(save_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(save_dict, f)
        
def load_model_info_for_nptk(filepath, load_keys=[]):
    with open(filepath, 'r') as f:
        load_dict = json.load(f)
    
    assert isinstance(load_keys, list)
    if len(load_keys) == 0:
        return load_dict
    
    custom_dict = dict()
    for _key in load_keys:
        custom_dict[_key] = load_dict[_key] if _key in load_dict else None
        
    return custom_dict
    

if __name__ == '__main__':

    model = get_example_model()
    args = OmegaConf.load("config/setting.yaml")

    check_graphmodule = convert_graphmodule_with_args(args, model)

    # ONNX
    batch_size = 1
    channel = 3
    width = 224
    height = 224

    convert_onnx = convert_onnx(args, model, batch_size, channel, width, height)
