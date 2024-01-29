# Transforms

Users can easily create their own augmentation recipe simply by listing their desired data transform modules. It's possible to adjust the intensity and frequency of each transform module, and the listed transform modules are applied in sequence to produce augmented data. In NetsPresso Trainer, a visualization tool is also provided through a Gradio demo, allowing users to see how their custom augmentation recipe produces the data for the model.

## Supporting transforms

The currently supported methods in NetsPresso Trainer are as follows. Since techniques are adapted from pre-existing codes, most of the parameters remain unchanged. We note that most of these parameter descriptions are derived from original implementations.

We appreciate all the original code owners and we also do our best to make other values.

### CenterCrop

This augmentation follows the [CenterCrop](https://pytorch.org/vision/0.15/generated/torchvision.transforms.CenterCrop.html) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "centercrop" to use `CenterCrop` transform. |
| `size` | (int or list) Desired output size of the crop. If size is an int, a square crop (size, size) is made. If provided a list of length 1, it will be interpreted as (size[0], size[0]). If a list of length 2 is provided, a square crop (size[0], size[1]) is made. |

<details>
  <summary>CenterCrop example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: centercrop
        size: 224
  ```
</details>

### ColorJitter

This augmentation follows the [ColorJitter](https://pytorch.org/vision/0.15/generated/torchvision.transforms.ColorJitter.html?highlight=colorjitter#torchvision.transforms.ColorJitter) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "colorjitter" to use `ColorJitter` transform. |
| `brightness` | (float or list) The brightness scale value is randomly selected within [max(0, 1 - brightness), 1 + brightness] or given [min, max] range. |
| `contrast` | (float or list) The contrast scale value is randomly selected within [max(0, 1 - contrast), 1 + contrast] or given [min, max] range. |
| `saturation` | (float or list) The saturation scale value is randomly selected within [max(0, 1 - saturation), 1 + saturation] or given [min, max] range. |
| `hue` | (float or list) The hue scale value is randomly selected within [max(0, 1 - hue), 1 + hue] or given [min, max] range. |
| `p` | (float) The probability of applying the color jitter. If set to `1.0`, the color transform is always applied. |

<details>
  <summary>ColorJitter example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: colorjitter
        brightness: 0.25
        contrast: 0.25
        saturation: 0.25
        hue: 0.1
        p: 1.0
  ```
</details>

### Mixing

We defined Mixing transform as the combination of CutMix and MixUp augmentation. This shuffles samples within a batch instead of processing per image. Therefore, **Mixing transform must be in the last function of augmentation racipe** if user wants to use it. Also, Mixing not assumes a batch size 1. If both MixUp and CutMix are activated, only one of two is randomly selected and used per batch processing.

Cutmix augmentation is based on [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) and MixUp augmentation is based on [mixup: Beyond empirical risk minimization](https://arxiv.org/pdf/1710.09412.pdf%C2%A0). These implementation follow the [RandomCutmix and RandomMixup](https://github.com/apple/ml-cvnets/blob/77717569ab4a852614dae01f010b32b820cb33bb/data/transforms/image_torch.py) in the ml-cvnets library.

Currently, NetsPresso Trainer does not support a Gradio demo visualization for Mixing. This feature is planned to be added soon.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "cutmix" to use `RandomCutmix` mix transform. |
| `mixup` | (list[float], optional) List of length 2 which contains [mixup alpha, applying probability]. If None, mixup is not applied. |
| `cutmix` | (list[float], optional) List of length 2 which contains [cutmix alpha, applying probability]. If None, cutmix is not applied. |
| `inplace` | (bool) Whether to operate as inplace. |

<details>
  <summary>Mixing example</summary>

  ```yaml
  augmentation:
    train:
      -
        name: mixing
        mixup: [0.25, 1.0]
        cutmix: ~
        inplace: false
  ```
</details>


### Pad

Pad an image with constant. This augmentation is based on the [Pad](https://pytorch.org/vision/0.15/generated/torchvision.transforms.Pad.html#torchvision.transforms.Pad) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "pad" to use `Pad` transform. |
| `size` | (int or list) Padding on each border. If a single int is provided, target size is (`size`, `size`). If a list is provided, it must be length 2, and will produce size of (`size[0]`, `size[1]`) padded image. If each edge of input image is greater or equal than target size, padding will be not processed. |
| `fill` | (int or list) If a single int is provided this is used to fill pixels with constant value. If a list of length 3, it is used to fill R, G, B channels respectively. |

<details>
  <summary>Pad example - 1</summary>
  
  ```yaml
  augmentation:
    train:
      -
        name: pad
        size: 512
        fill: 0
  ```
</details>

<details>
  <summary>Pad example - 2</summary>
  
  ```yaml
  augmentation:
    train:
      -
        name: pad
        size: [512, 512]
        fill: 0
  ```
</details>

### RandomCrop

Crop the given image at a random location. This augmentation follows the [RandomCrop](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomcrop" to use `RandomCrop` transform. |
| `size` | (int or list) Desired output size of the crop. If size is an int, a square crop (size, size) is made. If provided a list of length 1, it will be interpreted as (size[0], size[0]). If a list of length 2 is provided, a square crop (size[0], size[1]) is made. |

<details>
  <summary>RandomCrop example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomcrop
        size: 256
  ```
</details>

### RancomErasing

Erase random area of given image. This augmentation follows the [RandomErasing](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomErasing.html#torchvision.transforms.RandomErasing) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomerasing" to use `RancomErasing` transform. |
| `p` | (float) The probability of applying random erasing. If `1.0`, it always applies. |
| `scale` | (list) Range of proportion of erased area against input image. |
| `ratio` | (list) Range of aspect ratio of erased area. |
| `value` | (int, optional) Erasing value. If `None`, erase image with random noise. |
| `inplace` | (bool) Whether to operate as inplace. |

<details>
  <summary>RandomErasing example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomerasing
        p: 0.5
        scale: [0.02, 0.33]
        ratio: [0.3, 3.3]
        value: 0
        inplace: False
  ```
</details>

### RandomHorizontalFlip

Horizontally flip the given image randomly with a given probability. This augmentation follows the [RandomHorizontalFlip](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomhorizontalflip" to use `RandomHorizontalFlip` transform. |
| `p` | (float) the probability of applying horizontal flip. If `1.0`, it always applies the flip. |

<details>
  <summary>RandomHorizontalFlip example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomhorizontalflip
        p: 0.5
  ```
</details>

### RandomResizedCrop

Crop a random portion of image with different aspect of ratio in width and height, and resize it to a given size. This augmentation follows the [RandomResizedCrop](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomResizedCrop.html#torchvision.transforms.RandomResizedCrop) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomresizedcrop" to use `RandomResizedCrop` transform. |
| `size` | (int or list) Desired output size of the crop. If size is an int, a square crop (`size`, `size`) is made. If provided a list of length 1, it will be interpreted as (`size[0]`, `size[0]`). If a list of length 2 is provided, a crop with size (`size[0]`, `size[1]`) is made. |
| `scale` | (float or list) Specifies the lower and upper bounds for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image. |
| `ratio` | (float or list) lower and upper bounds for the random aspect ratio of the crop, before resizing. |
| `interpolation` | (str) Desired interpolation type. Supporting interpolations are 'nearest', 'bilinear' and 'bicubic'. |

<details>
  <summary>RandomResizedCrop</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomresizedcrop
        size: 256
        scale: [0.08, 1.0]
        ratio: [0.75, 1.33]
        interpolation: 'bilinear'
  ```
</details>


### RandomVerticalFlip

Vertically flip the given image randomly with a given probability. This augmentation follows the [RandomVerticalFlip](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomVerticalFlip.html#torchvision.transforms.RandomVerticalFlip) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomverticalflip" to use `RandomVerticalFlip` transform. |
| `p` | (float) the probability of applying vertical flip. If `1.0`, it always applies the flip. |

<details>
  <summary>RandomVerticalFlip example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomverticalflip
        p: 0.5
  ```
</details>

### Resize

Naively resize the input image to the given size. This augmentation follows the [Resize](https://pytorch.org/vision/0.15/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "resize" to use `Resize` transform. |
| `size` | (int or list) Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller or larger edge of the image will be matched to this number and keep aspect ratio. Determining match to smaller or larger edge is determined by `resize_criteria`. |
| `interpolation` | (str) Desired interpolation type. Supporting interpolations are 'nearest', 'bilinear' and 'bicubic'. |
| `max_size` | (int, optional) The maximum allowed for the longer edge of the resized image: if the longer edge of the image exceeds `max_size` after being resized according to `size`, then the image is resized again so that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller edge may be shorter than `size`. This is only supported if `size` is an int. |
| `resize_criteria` | (str, optional) This field only used when `size` is int. This determines which side (shorter or longer) to match with `size`, and only can have 'short' or 'long' or `None`. i.e, if `resize_criteria` is 'short' and height > width, then image will be rescaled to (size * height / width, size). |

<details>
  <summary>Resize example - 1</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: resize
        size: [256, 256]
        interpolation: 'bilinear'
        max_size: ~
        resize_criteria: ~
  ```
</details>

<details>
  <summary>Resize example - 2</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: resize
        size: 256
        interpolation: 'bilinear'
        max_size: ~
        resize_criteria: long
  ```
</details>

### TrivialAugmentWide

TrivialAugment based on [TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.pdf). This augmentation follows the [TrivialAugmentWide](https://pytorch.org/vision/0.15/generated/torchvision.transforms.TrivialAugmentWide.html#torchvision.transforms.TrivialAugmentWide) in the torchvision library. Currently, this transform function does not support segmentation and detection data.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "trivialaugmentwide" to use `TrivialAugmentWide` transform. |
| `num_magnitude_bins` | (int) The number of different magnitude values. |
| `interpolation` | (str) Desired interpolation type. Supporting interpolations are 'nearest', 'bilinear' and 'bicubic'. |
| `fill` | (list or int, optional) Pixel fill value for the area outside the transformed image. If given a number, the value is used for all bands respectively. |

<details>
  <summary>TrivialAugmentWide example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: trivialaugmentwide
        num_magnitude_bins: 31
        interpolation: 'bilinear'
        fill: ~
  ```
</details>
