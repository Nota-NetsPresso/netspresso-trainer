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

### HSVJitter

HSVJitter is based on `augment_hsv` function of [YOLOX repository](https://github.com/MegviiBaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/data/data_augment.py#L21-L31). This transform convert input image to HSV format, and randomly adjust according to magnitude configuration. Each channel can be randomly adjusted or remain unchanged for every transform step.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "hsvjitter" to use `HSVJitter` transform. |
| `h_mag` | (int) Randomly adjust the H channel within the range of [-h_mag, h_mag]. |
| `s_mag` | (int) Randomly adjust the S channel within the range of [-s_mag, s_mag]. |
| `v_mag` | (int) Randomly adjust the V channel within the range of [-v_mag, v_mag]. |

<details>
  <summary>HSVJitter example</summary>

  ```yaml
  augmentation:
    train:
      -
        name: hsvjitter
        h_mag: 5
        s_mag: 30
        v_mag: 30
  ```
</details>

### Mixing

We defined Mixing transform as the combination of CutMix and MixUp augmentation. This shuffles samples within a batch instead of processing per image. Therefore, **Mixing transform must be in the last function of augmentation racipe** if user wants to use it. Also, Mixing not assumes a batch size 1. If both MixUp and CutMix are activated, only one of two is randomly selected and used per batch processing.

Cutmix augmentation is based on [CutMix: Regularization strategy to train strong classifiers with localizable features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) and MixUp augmentation is based on [mixup: Beyond empirical risk minimization](https://arxiv.org/pdf/1710.09412.pdf%C2%A0). These implementation follow the [RandomCutmix and RandomMixup](https://github.com/apple/ml-cvnets/blob/77717569ab4a852614dae01f010b32b820cb33bb/data/transforms/image_torch.py) in the ml-cvnets library.

Currently, NetsPresso Trainer does not support a Gradio demo visualization for Mixing. This feature is planned to be added soon.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "mixing" to use `Mixing` transform. |
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


### MosaicDetection

This MosaicDetection augmentation is based on [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX). For each sample, the following steps are taken to create an augmented sample.

- Load three additional images.
- Resize the four images to fit the `size`.
- Merge the four images into one, with the merge center point randomly determined.
- Apply a random affine transformation to the merged image. And resize output image to fit the `size`.
- Finally, if `enable_mixup` is set to `True`, apply a mixup transformation with a fixed alpha of 0.5. The mixup image also is randomly loaded from dataset.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "mosaicdetection" to use `MosaicDetection` transform. |
| `size` | (list) Desired output size of the `MosaicDetection`. |
| `mosaic_prob` | (float) The probability of applying the `MosaicDetection`. If set to 1.0, it is always applied. |
| `affine_scale` | (list) Generate affine matrix with a scale range of [affine_scale[0], affine_scale[1]]. |
| `degrees` | (float) Generate affine matrix with a rotation range of [-degrees, degrees]. |
| `translate` | (float) Generate affine matrix with a translate range of [-translate, translate]. |
| `shear` | (float) Generate affine matrix with a shear range of [-shear, shear]. Randomly generate for each x-axis and y-axis. |
| `enable_mixup` | (bool) Whether to apply mixup. |
| `mixup_prob` | (float) The probability of applying the mixup. If set to 1.0, it is always applied. |
| `mixup_scale` | (list) Resize scale range for mixup image.  |
| `fill` | (int)  This is used to fill pixels with constant value. |
| `mosaic_off_duration` | (int) Number of epochs for which the `MosaicDetection` transform is disabled at the end of training. |

<details>
  <summary>MosaicDetection example</summary>

  ```yaml
  augmentation:
    train:
      -
        name: mosaicdetection
        size: [*img_size, *img_size]
        mosaic_prob: 1.0
        affine_scale: [0.5, 1.5]
        degrees: 10.0
        translate: 0.1
        shear: 2.0
        enable_mixup: True
        mixup_prob: 1.0
        mixup_scale: [0.5, 1.5]
        fill: 114
        mosaic_off_duration: 10
  ```
</details>

### Nomalize

Apply z-normalization to an image.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "normalize" to use `Normalize` transform. |
| `mean` | (list[float]) The mean values for normalizing the image. |
| `std` | (list[float]) The standard deviation values for normalizing the image. |

<details>
  <summary>Normalize example</summary>

  ```yaml
  augmentation:
    train:
      -
        name: normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
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

### PoseTopDownAffine

Apply affine transform based on given bounding box. This augmentation is based on the [RandomBBoxTransform](https://github.com/open-mmlab/mmpose/blob/5a3be9451bdfdad2053a90dc1199e3ff1ea1a409/mmpose/datasets/transforms/common_transforms.py) and [TopDownAffine](https://github.com/open-mmlab/mmpose/blob/5a3be9451bdfdad2053a90dc1199e3ff1ea1a409/mmpose/datasets/transforms/topdown_transforms.py) in mmpose library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "pad" to use `Pad` transform. |
| `scale` | (list) Randomly adjust box scale in range of [`scale[0]`, `scale[1]`] |
| `scale_prob` | (float) The probability of applying scaling. If set to `1.0`, scale of box always randomly adjusted. |
| `translate` | (float) Randomly adjust the offset of the box by adding translate factor * box size. The translate factor is random value in range of [`0`, `translate`]. |
| `translate_prob` | (float) The probability of applying translate. If set to `1.0`, offset of box always randomly adjusted. |
| `rotation` | (int) Random rotation range of affine transform. The random value determined in [`-rotation`, `rotation`]. This rotation angle is degree. |
| `rotation_prob` | (float) The probability of applying rotation. If set to `1.0`, affine transform matrix always contains rotation value. |

<details>
  <summary>PoseTopDownAffine example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: posetopdownaffine
        scale: [0.75, 1.25]
        scale_prob: 1.
        translate: 0.1
        translate_prob: 1.
        rotation: 60
        rotation_prob: 1.
        size: [*img_size, *img_size]
  ```
</details>

### RandomCrop

Crop the given image at a random location. This augmentation follows the [RandomCrop](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop) in torchvision library.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomcrop" to use `RandomCrop` transform. |
| `size` | (int or list) Desired output size of the crop. If size is an int, a square crop (size, size) is made. If provided a list of length 1, it will be interpreted as (size[0], size[0]). If a list of length 2 is provided, a square crop (size[0], size[1]) is made. |
| `fill` | (int or list) If a single int is provided this is used to fill pixels with constant value. If a list of length 3, it is used to fill R, G, B channels respectively.
<details>
  <summary>RandomCrop example</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomcrop
        size: 256
        fill: 114
  ```
</details>

### RandomErasing

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

### RandomResize

RandomResize transforms the input image to a random size within a specified range. This random size range is determined by [`base_size[0]` - `stride` * `v`, `base_size[1]` + `stride` * `v`] with `stride` interval, where the value `v` is an integer within the range of [`-random_range`, `random_range`]. E.g. If `base_size = [256, 256]`, `stride = 32`, `random_range = 2`, possible output image sizes are `[[192, 192], [224, 224], [256, 256], [288, 288], [320, 320]]`.

Since applying random resize to every image arises the difficulty of a batch handling, random resize should be applied on a per-batch basis. However, due to current implementation constraints, it's challenging to apply randomness at the batch level, so a single random size is determined per dataloader worker. The size managed by each worker changes to at the start of each epoch. Therefore, to fully benefit from RandomResize, the number of workers set by `environment.num_workers` needs to be sufficiently large.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomresize" to use `RandomResize` transform. |
| `base_size` | (list) The base size of the output image after random resizing. The output size is determined based on `base_size`. |
| `stride` | (int) The interval at which the size variation occurs. |
| `random_range` | (int) The range for random size variation. The final size is determined in range of [`base_size[0]` - `stride` * `v`, `base_size[1]` + `stride` * `v`] with `stride` interval, where `v` is an integer within the range of [`-random_range`, `random_range`]. |
| `interpolation` | (str) Desired interpolation type. Supporting interpolations are 'nearest', 'bilinear' and 'bicubic'. |

<details>
  <summary>RandomResize</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomresize
        base_size: [256, 256]
        stride: 32
        random_range: 4
        interpolation: 'bilinear'
  ```
</details>


### RandomResize2

RandomResize2 transforms the input image by resizing it based on a randomly selected scaling factor within a specified range. Note that RandomResize2 preserves the original aspect ratio, result image size might be largely different with `base_size`.

Applying random resize to every image arises the difficulty of a batch handling, and RandomResize2 does not support per-batch target size handling function. We recommend to use RandomResize2 with other trasnform method.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "randomresize" to use `RandomResize` transform. |
| `base_size` | (list) The base (height, width) of the target image, which is the initial size before applying randomness. |
| `random_range` | (int) A range [min_factor, max_factor] within which the random scaling factor is selected. The input image will be resized by a combination of base_size and random factors, while maintaining the aspect ratio. |
| `interpolation` | (str) Desired interpolation type. Supporting interpolations are 'nearest', 'bilinear' and 'bicubic'. |

<details>
  <summary>RandomResize</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: randomresize
        base_size: [512, 2048]
        random_range: [0.5, 1.5]
        interpolation: 'bilinear'
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

### ToTensor

The `ToTensor` transform converts data into a tensor that can be fed into a PyTorch model. The `pixel_range` parameter allows you to specify the range of pixel values that each image data point can have.

| Field <img width=200/> | Description |
|---|---|
| `name` | (str) Name must be "totensor" to use `ToTensor` transform. |
| `pixel_range` | (float) The range of pixel values that the image data will be normalized to. |

<details>
  <summary>ToTensor example - 1</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: totensor
        pixel_range: 1.0
  ```
</details>

<details>
  <summary>ToTensor example - 2</summary>
  
  ```yaml
  augmentation:
    train:
      - 
        name: totensor
        pixel_range: 255.0
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
