<!-- FIXME: mostly copied from https://github.com/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/blob/main/models/torch/README.md -->

### PyTorch GraphModule with fx tracer

To fully use with PyNetsPresso, the model checkpoint from trainer should be converted to symbolic traced format. Thanks to PyTorch team, [torch FX] is provided as a toolkit for developers to transform `nn.Module` with symbolic tracing, graph representation, and python code generation. Please refer to [torch FX] docuement for more details.

Our trainer provides the trained model checkpoint with both onnx format (`.onnx`) and graphmodule format (`.pt`). So, you don't have to convert the model manually after training.
The followings are several code blocks about loading and converting FX graph.

#### GraphModule convert

```python
import torch.fx
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
graph = torch.fx.Tracer().trace(model)
traced_model = torch.fx.GraphModule(model, graph)
torch.save(traced_model, "resnet18.pt")
```

#### GrahModule inference

```python
import torch
import torch.fx
import numpy as np

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
graph = torch.fx.Tracer().trace(model)
traced_model = torch.fx.GraphModule(model, graph)

# input size is needed to be choosen
input_shape = (1, 3, 224, 224)
random_input = torch.Tensor(np.random.randn(*input_shape))

with torch.no_grad():
    original_output = model(random_input)
    traced_output = traced_model(random_input)

assert torch.allclose(original_output, traced_output), "inference result is not equal!"
```

#### Load GraphModule

```python
import torch

model = torch.load("resnet18.pt")

```


- [torch FX]: https://pytorch.org/docs/1.12/fx.html
