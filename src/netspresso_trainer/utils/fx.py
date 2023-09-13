import torch
import torch.fx as fx
import torch.nn as nn

__all__ = ['convert_graphmodule', 'save_graphmodule']

def _convert_graphmodule(model: nn.Module) -> fx.GraphModule:
    try:
        _graph = fx.Tracer().trace(model)
        model = fx.GraphModule(model, _graph)
        return model
    except Exception as e:
        raise e


def convert_graphmodule(model: nn.Module) -> fx.GraphModule:
    return _convert_graphmodule(model)


def save_graphmodule(model: nn.Module, f):
    model: fx.GraphModule = _convert_graphmodule(model)
    torch.save(model, f)
