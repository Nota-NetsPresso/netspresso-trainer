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

- AdamW
- Adam
- Adadelta
- Adagrad
- RMSprop
- Adamax
- SGD

If you are unsure which optimizer to use, we recommend reading the [blog post from *towardsdatascience*](https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e).

## Learning rate scheduler

NetsPresso Trainer supports various learning rate schedulers based on PyTorch.
In particular, warm-up is supported for frequently used learning rate schedulers, and warm restart is supported for some schedulers, such as cosine annealing.

### Supporting schedulers

## Gradio demo for simulating the learning rate scheduler

In many training feature repositories, it is recommended to perform the entire training pipeline and check the log to see how the learning rate scheduler works.
NetsPresso Trainer supports learning rate schedule simulation to allow users to easily understand the learning rate scheduler for their configured training recipe.
By copying the training configuration into the simulator, users can see how the learning rate changes every epoch.

> :warning: This simulation is not supported for some schedulers which adjust the learning rate dynamically with training results.

#### Running on your environment
*FIXME*

#### Hugging Face Spaces
*FIXME*

## Field list