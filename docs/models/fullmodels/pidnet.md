# PIDNet

PIDNet model based on [PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers](https://arxiv.org/abs/2206.02066).

## Field list

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "pidnet" to use PIDNet model. |
| `m` | (int) Residual block repetetion count for 1, 2 stage of I branch, and 3, 4 stage of P branch. |
| `n` | (int) Residual block repetition count for 3, 4 stage of I branch |
| `channels` | (int) Base dimension of the overall model except ppm and head module. |
| `ppm_channels` | (int) Dimension of the ppm module. |
| `head_channels` | (int) Dimension of the head module. |

<details>
  <summary>PIDNet-s</summary>
  
  ```yaml
  model:
    architecture:
      full:
        name: pidnet
        m: 2
        n: 3
        channels: 32
        ppm_channels: 96
        head_channels: 128
  ```
</details>

<details>
  <summary>PIDNet-m</summary>
  
  ```yaml
  model:
    architecture:
      full:
        name: pidnet
        m: 2
        n: 3
        channels: 64
        ppm_channels: 96
        head_channels: 128
  ```
</details>

<details>
  <summary>PIDNet-l</summary>
  
  ```yaml
  model:
    architecture:
      full:
        name: pidnet
        m: 3
        n: 4
        channels: 64
        ppm_channels: 112
        head_channels: 256
  ```
</details>

## Related links
- [`XuJiacong/PIDNet`](https://github.com/XuJiacong/PIDNet)
