# Optimizers

NetsPresso Trainer uses the optimizers implemented in PyTorch as is. By selecting an optimizer that suits your training recipe, you can configure the optimal training. If you are unsure which optimizer to use, we recommend reading the [blog post from *towardsdatascience*](https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e).

## Supporting optimizers

The currently supported methods in NetsPresso Trainer are as follows. Since techniques are adapted from pre-existing codes, most of the parameters remain unchanged. We note that most of these parameter descriptions are derived from original implementations.

We appreciate all the original code owners and we also do our best to make other values.

### Adadelta

This optimizer follows the [Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "adadelta" to use `Adadelta` optimizer. |
| `lr` | (float) Coefficient that scales delta before it is applied to the parameters. |
| `rho` | (float) Coefficient used for computing a running average of squared gradients |
| `weight_decay` | (float) weight decay (L2 penalty). |

<details>
  <summary>Adadelta example</summary>
  ```yaml
  training:
    optimizer:
      name: adadelta
      lr: 1.0
      rho: 0.9
      weight_decay: 0.
  ```
</details>

### Adagrad

This optimizer follows the [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "adagrad" to use `Adagrad` optimizer. |
| `lr` | (float) Learning rate. |
| `lr_decay` | (float) Learning rate decay. |
| `weight_decay` | (float) weight decay (L2 penalty). |

<details>
  <summary>Adagrad example</summary>
```yaml
training:
  optimizer:
    name: adagrad
    lr: 1e-2
    lr_decay: 0.
    weight_decay: 0.
```
</details>

### Adam

This optimizer follows the [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) (`adam`) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "adam" to use `Adam` optimizer. |
| `lr` | (float) Learning rate. |
| `betas` | (float) Coefficients used for computing running averages of gradient and its square. |
| `weight_decay` | (float) weight decay (L2 penalty). |

<details>
  <summary>Adam example</summary>
```yaml
training:
  optimizer:
    name: adam
    lr: 1e-3
    betas: [0.9, 0.999]
    weight_decay: 0.
```
</details>

### Adamax

This optimizer follows the [Adamax](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "adamax" to use `Adamax` optimizer. |
| `lr` | (float) Learning rate. |
| `betas` | (float) Coefficients used for computing running averages of gradient and its square. |
| `weight_decay` | (float) weight decay (L2 penalty). |

<details>
  <summary>Adamax example</summary>
```yaml
training:
  optimizer:
    name: adamax
    lr: 2e-3
    betas: [0.9, 0.999]
    weight_decay: 0.
```
</details>

### AdamW

This optimizer follows the [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "adamw" to use `AdamW` optimizer. |
| `lr` | (float) Learning rate. |
| `betas` | (list[float]) Coefficients used for computing running averages of gradient and its square. |
| `weight_decay` | (float) weight decay (L2 penalty). |

<details>
  <summary>AdamW example</summary>
```yaml
training:
  optimizer:
    name: adamw
    lr: 1e-3
    betas: [0.9, 0.999]
    weight_decay: 0.
```
</details>

### RMSprop

This optimizer follows the [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "rmsprop" to use `RMSprop` optimizer. |
| `lr` | (float) Learning rate. |
| `alpha` | (float) Smoothing constant. |
| `momentum` | (float) Momentum factor. |
| `weight_decay` | (float) weight decay (L2 penalty). |
| `eps` | (float) Term added to the denominator to improve numerical stability. |

<details>
  <summary>RMSprop example</summary>
```yaml
training:
  optimizer:
    name: rmsprop
    lr: 1e-2
    alpha: 0.99
    momentum: 0.
    weight_decay: 0.
    eps: 1e-8
```
</details>

### SGD

This optimizer follows the [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "sgd" to use `SGD` optimizer. |
| `lr` | (float) Learning rate. |
| `momentum` | (float) Momentum factor. |
| `weight_decay` | (float) weight decay (L2 penalty). |
| `nesterov` | (bool) Enables Nesterov momentum. |

<details>
  <summary>SGD example</summary>
```yaml
training:
  optimizer:
    name: sgd
    lr: 1e-2
    momentum: 0.
    weight_decay: 0.
    nesterov: false
```
</details>
