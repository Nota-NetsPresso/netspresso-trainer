# Gradient Clipping

Gradient clipping is a technique used to mitigate the exploding gradient problem in training deep neural networks, particularly in scenarios involving recurrent neural networks (RNNs) or models with long dependency chains. 

During the training of neural networks, especially those with deep architectures or recurrent structures, gradients can sometimes grow exponentially large. This phenomenon, known as exploding gradients, can lead to several issues as follows:

- Numerical instability
- Overshooting optimal parameter values
- Divergence in the training process

Gradient clipping addresses this issue by limiting the magnitude of gradients during backpropagation. This technique ensures that gradient updates remain within a reasonable range, promoting more stable and controlled training.

### Norm Clipping
For now, netspresso-trainer supports the norm clipping. This method scales down the gradient when its norm exceeds a threshold $v$.
$$
\text{if } ||\mathbf{g}|| > v \text{ then } \mathbf{g} \leftarrow v \cdot \frac{\mathbf{g}}{||\mathbf{g}||}
$$

| Field <img width=200/> | Description |
|---|---|
| `training.max_norm` | (float) The norm threshold. For gradient clipping, this means the maximum gradient value. To disable the gradient clipping, you can set this value to None (~).|

<details>
  <summary>Norm gradient clipping example</summary>
```yaml
training:
    max_norm: 0.1
```
</details>