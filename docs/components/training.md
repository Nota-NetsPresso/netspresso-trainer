# Overview

In training, the training recipe is just as important as the model architecture. Even if you have a good model architecture, the performance on the same data and model combination can vary greatly depending on the training recipe.
NetsPresso Trainer not only introduces models optimized for edge devices, but also provides the ability to change training configurations to train these models with various data.
The optimal training recipe will vary depending on the data you want to train. Use the options provided by NetsPresso Trainer to find the best training recipe for your data.

## Batch size and epochs

The batch size and epoch used in training may vary depending on the GPU and server specifications you have. If the batch size is large, it tends to take up a lot of GPU memory, and as the number of epochs trained increases, it tends to take a long time to complete the training.
Adjust the values according to your server specifications, but for successful training, it is recommended to set the batch size to at least 8.

## Optimizers

NetsPresso Trainer uses the optimizers implemented in PyTorch as is. By selecting an optimizer suitable for the training recipe, including batch size, you can configure the optimal training.

### Supporting optimizers

- [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) (`adamw`)
- [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) (`adam`)
- [Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html) (`adadelta`)
- [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html) (`adagrad`)
- [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) (`rmsprop`)
- [Adamax](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html) (`adamax`)
- [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) (`sgd`)

If you are unsure which optimizer to use, we recommend reading the [blog post from *towardsdatascience*](https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e).

## Learning rate scheduler

NetsPresso Trainer supports various learning rate schedulers based on PyTorch.
In particular, warm-up is supported for frequently used learning rate schedulers, and warm restart is supported for some schedulers, such as cosine annealing.
NetsPresso Trainer updates the learning rate at the end of epoch, not the end of step, so users will set the scheduler with epoch-level counts.

### Supporting schedulers

## Gradio demo for simulating the learning rate scheduler

In many training feature repositories, it is recommended to perform the entire training pipeline and check the log to see how the learning rate scheduler works.
NetsPresso Trainer supports learning rate schedule simulation to allow users to easily understand the learning rate scheduler for their configured training recipe.
By copying the training configuration into the simulator, users can see how the learning rate changes every epoch.

> :warning: This simulation is not supported for some schedulers which adjust the learning rate dynamically with training results.

#### Running on your environment

Please run the gradio demo with following command:

```
python demo/gradio_lr_scheduler.py
```

#### Hugging Face Spaces

The example simulation will be able to use with Hugging Face Spaces at [nota-ai/netspresso-trainer-lr-scheduler](https://huggingface.co/spaces/nota-ai/netspresso-trainer-lr-scheduler).


## Field list

| Field <img width=200/> | Description |
|---|---|
| `training.seed` | (int) random seed |
| `training.opt` | (str) the type of optimizer. Please check [the list of supporting optimizer](#supporting-optimizers) for more details. |
| `training.lr` | (float) base learning rate |
| `training.momentum` | (float) momentum value for optimizer |
| `training.weight_decay` | (float) the strength of L2 penalty  |
| `training.sched` | (str) the type of scheduler. Please check [the list of supporting LR schedulers](#supporting-schedulers) for more details. |
| `training.min_lr` | (float) the minimum value of learning rate |
| `training.warmup_bias_lr` | (float) the starting learning rate of warmup period in learning rate scheduling |
| `training.warmup_epochs` | (int) the warmup period |
| `training.iters_per_phase` | (int) the period of base phase in learning rate scheduling. Applied when the scheduler is `step` or `cosine`. |
| `training.sched_power` | (float) the power value of polynomial scheduler. Applied when the scheduler is `poly`. |
| `training.epochs` | (int) the total number of epoch for training the model |
| `training.batch_size` | (int) the number of samples in single batch input |