# Normalizations

Normalization layers significantly affect model performance by transforming features to similar scales. To enable diverse model designs, NetsPresso Trainer supports various activation modules based on the pytorch library.

## Supporting normalization layers

The currently supported normalization layers in NetsPresso Trainer are as follows.

### BatchNorm
- This can be applied by giving `'batch_norm'` keyword
- Batch normalization follows [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) in the PyTorch library.

### InstanceNorm
- This can be applied by giving `'instance_norm'` keyword
- Instance normalization follows the [torch.nn.InstanceNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d) in the PyTorch library