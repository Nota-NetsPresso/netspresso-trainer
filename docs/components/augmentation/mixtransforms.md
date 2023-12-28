# Mix transforms

We have named the techniques that shuffle samples within a batch after the processing of transform modules as "Mix transforms". Therefore, the modules supported in mix transforms do not assume a batch size 1, and when multiple mix transforms are applied, only one random mix transform is used per batch processing. Users can employ desired mix transform modules by listing them in the configuration.

Currently, NetsPresso Trainer does not support a Gradio demo for visualizing mix transforms. This feature is planned to be added soon.

## Supporting transforms

The currently supported methods in NetsPresso Trainer are as follows. Since techniques are adapted from pre-existing codes, hence most of the parameters remain unchanged. We notice that most of these parameter descriptions are from original implementations.

We appreciate all the original code owners and we also do our best to make other values.

### RandomCutmix

Cutmix augmentation based on [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf). This augmentation follows the [RandomCutmix](https://github.com/apple/ml-cvnets/blob/77717569ab4a852614dae01f010b32b820cb33bb/data/transforms/image_torch.py) in the ml-cvnets library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "cutmix" to use `RandomCutmix` mix transform. |
| `alpha` | (float) Mixing strength alpha. |
| `p` | (float) the probability of applying cutmix. If `1.0`, it always applies. |
| `inplace` | (bool) Whether to operate as inplace. |

<details>
  <summary>Cutmix</summary>
  
  ```yaml
  augmentation:
    mix_transforms:
      -
        name: cutmix
        alpha: 1.0
        p: 1.0
        inplace: False
  ```
</details>

### RandomMixup

Mixup augmentation based on [mixup: Beyond empirical risk minimization](https://arxiv.org/pdf/1710.09412.pdf%C2%A0). This augmentation follows the [RandomMixup](https://github.com/apple/ml-cvnets/blob/77717569ab4a852614dae01f010b32b820cb33bb/data/transforms/image_torch.py) in the ml-cvnets library.


| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "mixup" to use `RandomMixup` mix transform. |
| `alpha` | (float) Mixing strength alpha. |
| `p` | (float) the probability of applying cutmix. If `1.0`, it always applies. |
| `inplace` | (bool) Whether to operate as inplace. |

<details>
  <summary>Cutmix + Mixup</summary>
  
  ```yaml
  augmentation:
    mix_transforms:
      -
        name: mixup
        alpha: 0.25
        p: 1.0
        inplace: False
  ```
</details>
