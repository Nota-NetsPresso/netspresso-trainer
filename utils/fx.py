import torch
import torch.fx as fx


def _convert_graphmodule(model):
    try:
        _graph = fx.Tracer().trace(model)
        model = fx.GraphModule(model, _graph)
        return model
    except Exception as e:
        raise e


def convert_graphmodule(model):
    return _convert_graphmodule(model)


def save_graphmodule(model, f):
    model = _convert_graphmodule(model)
    torch.save(model, f)
