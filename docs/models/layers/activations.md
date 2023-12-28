# Activations

Choosing Activation functions is a key part of neural network design, since they determines the output of nodes in the network. They introduce non-linear properties to the network, enabling it to learn complex data patterns and make more sophisticated predictions.

To enable diverse model designs, NetsPresso Trainer supports various activation modules based on the pytorch library.

## Supporting activation functions

The currently supported activation functions in NetsPresso Trainer are as follows.

### ReLU
- This can be applied by giving `'relu'` keyword
- ReLU activation follows the [torch.nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) in the PyTorch library.

### PReLU
- This can be applied by giving `'prelu'` keyword
- PReLU activation follows the [torch.nn.PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html#torch.nn.PReLU) in the PyTorch library.

### Leaky ReLU
- This can be applied by giving `'leaky_relu'` keyword
- Leaky ReLU activation follows the [torch.nn.LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU) in the PyTorch library.

### GELU
- This can be applied by giving `'gelu'` keyword
- GELU activation follows the [torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU) in the PyTorch library.

### SiLU
- This can be applied by giving `'silu'` or `'swish'`keyword
- SiLU activation follows the [torch.nn.SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU) in the PyTorch library.

### Hardswish
- This can be applied by giving `'hard_swish'` keyword
- Hardswish activation follows the [torch.nn.Hardswish](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html#torch.nn.Hardswish) in the PyTorch library.