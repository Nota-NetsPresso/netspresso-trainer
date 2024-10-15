# Schedulers

NetsPresso Trainer supports various learning rate schedulers based on PyTorch. In particular, learning rate warm-up is supported for frequently used schedulers, and learning rate restart is supported for some schedulers, such as cosine annealing. NetsPresso Trainer updates the learning rate at the end of epoch, not the end of step, so users will set the scheduler with epoch-level counts.

## Supporting schedulers

The currently supported methods in NetsPresso Trainer are as follows. Since techniques are adapted from pre-existing codes, most of the parameters remain unchanged. We note that most of these parameter descriptions are derived from original implementations.

We appreciate all the original code owners and we also do our best to make other values.

### Step

This scheduler follows the [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "step" to use `StepLR` scheduler. |
| `iters_per_phase` | (int) Epoch period of learning rate decay. |
| `gamma` | (float) Multiplicative factor of learning rate decay. |
| `end_epoch` | (int) End epoch of this scheduler. Remained epochs will be trained with fixed learning rate. |

<details>
  <summary>Step example</summary>
  ```yaml
  training:
    scheduler:
      name: step
      iters_per_phase: 1
      gamma: 0.1
      end_epoch: 80
  ```
</details>

### Polynomial with warmup

This scheduler follows the [PolynomialLR](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#PolynomialLR) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "poly" to use `PolynomialLRWithWarmUp` scheduler. |
| `warmup_epochs` | (int) The number of steps that the scheduler finishes to warmup the learning rate. |
| `warmup_bias_lr` | (float) Starting learning rate for warmup period. |
| `min_lr` | (float) Minimum learning rate. |
| `power` | (float) The power of the polynomial. |
| `end_epoch` | (int) End epoch of this scheduler. At the `end_epoch`, learning rate will be `min_lr`, and remained epochs trained with fixed learning rate. |

<details>
  <summary>Polynomial with warmup example</summary>
```yaml
training:
  scheduler:
    name: poly
    warmup_epochs: 5
    warmup_bias_lr: 1e-5
    min_lr: 1e-6
    power: 1.0
    end_epoch: 80
```
</details>

### Cosine annealing with warmup

This scheduler follows the [CosineAnnealingLR](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "cosine_no_sgdr" to use `CosineAnnealingLRWithCustomWarmUp` scheduler. |
| `warmup_epochs` | (int) The number of steps that the scheduler finishes to warmup the learning rate. |
| `warmup_bias_lr` | (float) Starting learning rate for warmup period. |
| `min_lr` | (float) Minimum learning rate. |
| `end_epoch` | (int) End epoch of this scheduler. At the `end_epoch`, learning rate will be `min_lr`, and remained epochs trained with fixed learning rate. |

<details>
  <summary>Cosine annealing with warmup example</summary>
```yaml
training:
  scheduler:
    name: cosine_no_sgdr
    warmup_epochs: 5
    warmup_bias_lr: 1e-5
    min_lr: 1e-6
    end_epoch: 80
```
</details>

### Cosine annealing warm restarts with warmup

This scheduler follows the [CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "cosine" to use `CosineAnnealingWarmRestartsWithCustomWarmUp` scheduler. |
| `warmup_epochs` | (int) The number of steps that the scheduler finishes to warmup the learning rate. |
| `warmup_bias_lr` | (float) Starting learning rate for warmup period. |
| `min_lr` | (float) Minimum learning rate. |
| `iters_per_phase` | (float) Epoch period for the learning rate restart. |

<details>
  <summary>Cosine annealing warm restart with warmup example</summary>
```yaml
training:
  scheduler:
    name: cosine
    warmup_epochs: 5
    warmup_bias_lr: 1e-5
    min_lr: 1e-6
    iters_per_phase: 10
```
</details>

### Multi step

This scheduler follows the [MultiStepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html) in torch library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "multi_step" to use `MultiStepLR` scheduler. |
| `milestones` | (list) List of epoch indices. Must be increasing. |
| `gamma` | (float) Multiplicative factor of learning rate decay. |

<details>
  <summary>Step example</summary>
```yaml
training:
  scheduler:
    name: multi_step
    milestones: [30, 80]
    gamma: 0.1
```
</details>


## Gradio demo for simulating the learning rate scheduler

In many training feature repositories, it is recommended to perform the entire training pipeline and check the log to see how the learning rate scheduler works.
NetsPresso Trainer supports learning rate schedule simulation to allow users to easily understand the learning rate scheduler for their configured training recipe.
By copying and pasting the training configuration into the simulator, users can see how the learning rate changes every epoch.

> :warning: This simulation is not supported for some schedulers which adjust the learning rate dynamically with training results.

#### Running on your environment

Please run the gradio demo with following command:

```bash
bash scripts/run_simulator_lr_scheduler.sh
```