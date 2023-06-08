import torch
import torch.nn as nn
import torch.fx as fx

from models.backbones.experimental.resnet import resnet50


def convert_graphmodule(model):
    try:
        _graph = fx.Tracer().trace(model)
        model = fx.GraphModule(model, _graph)
        return model
    except Exception as e:
        raise e


if __name__ == '__main__':
    model = resnet50(task='')
    fx_model = convert_graphmodule(model)
    print(fx_model)
