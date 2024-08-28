# EMA (Exponential Moving Average)

In many cases, providing a model with averaged parameters brings performance benefits. The Exponential Moving Average (EMA) model is updated after each batch training step according to the following:

```python
ema_param = decay * ema_param + (1. - decay) * training_model_param
```

If EMA is enabled, both validation and model saving are processed with EMA model. Note that after the validation phase, the training model parameters are reverted back to the non-averaged model.

## EMA decay schedulers

It is often benefits to start with a smaller decay value at the beginning of training and gradually use higher values as progresses. To this, we support some decay scheduling methods.

### Constant decay

Constant decay keeps the decay value unchanged throughout the entire training process.

| Field <img width=200/> | Description |
|---|---|
| `training.epochs` | (str) Name must be "constant_decay" to use constant decay. |
| `training.decay` | (float) The decay rate for EMA. Its range must be in [0, 1.0]. If `None`. |

<details>
  <summary>Constant decay example</summary>
```yaml
training:
  ema:
    name: constant_decay
    decay: 0.9999
```
</details>

### Exponential decay

Exponential decay increases the decay value exponentially with the number of updates as following:

```python
applied_decay = decay * (1 - math.exp(-counter / beta)
```

`decay` and `beta` from configuration determine the maximum value of decay and the speed of convergence, respectively. The `counter` starts at 0 and increments by 1 with each update.

| Field <img width=200/> | Description |
|---|---|
| `training.name` | (str) Name must be "exp_decay" to use constant decay. |
| `training.decay` | (float) The decay rate for EMA. For exponential decay, this means maximum decay value. Its range must be in [0, 1.0]. |
| `training.beta` | (float) Determines the speed of convergence of decay to maximum value. |

<details>
  <summary>Exponential decay example</summary>
```yaml
training:
  ema:
    name: exp_decay
    decay: 0.9999
    beta: 100
```
</details>