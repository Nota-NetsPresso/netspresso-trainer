# Overview

In training, the training recipe is just as important as the model architecture. Even if you have a good model architecture, the performance on the same data and model combination can vary greatly depending on your training recipe. NetsPresso Trainer not only introduces models optimized for edge devices, but also provides the ability to change training configurations to train these models with various data. The optimal training recipe will vary depending on the data you want to train. Use the options provided by NetsPresso Trainer to find the optimal training recipe for your data.

Users can adjust epochs, the desired optimizer and scheduler as a following example.

```yaml
training:
  epochs: 300
  ema:
    name: constant_decay
    decay: 0.9999
  optimizer:
    name: adamw
    lr: 6e-5
    betas: [0.9, 0.999]
    weight_decay: 0.0005
  scheduler:
    name: cosine_no_sgdr
    warmup_epochs: 5
    warmup_bias_lr: 1e-5
    min_lr: 0.
```

## Field list

| Field <img width=200/> | Description |
|---|---|
| `training.epochs` | (int) The total number of epoch for training the model |
| `training.ema` | (dict, optional) The configuration of EMA. Please refer to [the EMA page](./ema.md) for more details. If `None`, EMA is not applied. |
| `training.optimizer` | (dict) The configuration of optimizer. Please refer to [the list of supporting optimizer](./optimizers.md) for more details. |
| `training.scheduler` | (dict) The configuration of learning rate scheduler. Please refer to [the list of supporting scheduler](./schedulers.md) for more details. |