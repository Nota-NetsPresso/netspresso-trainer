# Augmentation

NetsPresso Trainer provides data augmentation function to improve model performance, allowing users to configure their own training recipes as desired. 
Data augmentation in NetsPresso Trainer is based on torch and torchvision, and all augmentations are written based on `pillow` images and `torch.Tensor`.  

Currently, there is no function for users to directly implement augmentation recipes, but only customize variable values to adjust the intensity and frequency of augmentations. In the near future, a function to directly design augmentation recipes will be added. 

## Supporting transforms

- The currently supported augmentation methods in NetsPresso Trainer are as follows.

### ColorJitter

- This augmentation follows the [ColorJitter](https://pytorch.org/vision/0.15/generated/torchvision.transforms.ColorJitter.html?highlight=colorjitter#torchvision.transforms.ColorJitter) in torchvision library.

### Identity

- Identity passing. No changes in data.

### Normalize

- Normalize the value to follow the distribution. It is mostly used in converting from pillow image to `torch.Tensor`.

### Pad

- Pad an image. This augmentation follows the [Pad](https://pytorch.org/vision/0.15/generated/torchvision.transforms.Pad.html#torchvision.transforms.Pad) in torchvision library.

### PadIfNeeded

- Relatively pad an image to get the final image with the given size. This augmentation follows the [PadIfNeeded](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded) in albumentations library.

### RandomCrop

- Crop the given image at a random location. This augmentation follows the [RandomCrop](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop) in torchvision library.

### RandomHorizontalFlip

- Horizontally flip the given image randomly with a given probability. This augmentation follows the [RandomHorizontalFlip](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip) in torchvision library.

### RandomResizedCrop

- Crop a random portion of image with different aspect of ratio in width and height, and resize it to a given size. This augmentation follows the [RandomResizedCrop](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomResizedCrop.html#torchvision.transforms.RandomResizedCrop) in torchvision library.

### RandomVerticalFlip

- Vertically flip the given image randomly with a given probability. This augmentation follows the [RandomVerticalFlip](https://pytorch.org/vision/0.15/generated/torchvision.transforms.RandomVerticalFlip.html#torchvision.transforms.RandomVerticalFlip) in torchvision library.

### Resize

- Naively resize the input image to the given size. This augmentation follows the [Resize](https://pytorch.org/vision/0.15/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize) in torchvision library.

### ToTensor

- Convert from pillow image to `torch.Tensor` by transposing dimensions and changing `dtype` from `uint8` to `float32` if needed.


## Gradio demo for simulating the transform

In many learning function repositories, it is recommended to check the code and documentation for augmentations or actually run the training to check the logs to see how augmentations are performed. 
NetsPresso Trainer supports augmentation simulation to help users easily understand the augmentation recipe they have configured. 
By copying and entering the augmentation configuration into the simulator, users can preview how a specific image will be augmented in advance. However, transforms (e.g. Normalize, ToTensor) used to convert the image array for learning purposes are excluded from the simulation visualization process. 
In particular, since this simulator directly imports the augmentation modules used in NetsPresso Trainer, users can use the same functions as the augmentation functions used in actual training to verify the results.  

Our team hopes that the learning process with NetsPresso Trainer will become a more enjoyable experience for all users. 

### How to use

#### Running on your environment

Please run the gradio demo with following command:

```
bash scripts/run_simulator_augmentation.sh
```

#### Hugging Face Spaces

The example simulation will be able to use with Hugging Face Spaces at [nota-ai/netspresso-trainer-augmentation](https://huggingface.co/spaces/nota-ai/netspresso-trainer-augmentation).

## Field list

### Common

| Field <img width=200/> | Description |
|---|---|
| `augmentation.img_size` | (int) the image size of model input after finishing the data augmentation |

### Resize

| Field <img width=200/> | Description |
|---|---|
| `augmentation.crop_size_h` | (int) the height of cropped image |
| `augmentation.crop_size_w` | (int) the width of cropped image |

### RandomResizedCrop

| Field <img width=200/> | Description |
|---|---|
| `augmentation.crop_size_h` | (int) the height of cropped image |
| `augmentation.crop_size_w` | (int) the width of cropped image |
| `augmentation.resize_ratio0` | (float) the minimum scale of random image resizing |
| `augmentation.resize_ratiof` | (float) the maximum scale of random image resizing |

### RandomHorizontalFlip

| Field <img width=200/> | Description |
|---|---|
| `augmentation.fliplr` | (float) the probability of the flip. If `1.0`, it always flips the image. |

### ColorJitter

| Field <img width=200/> | Description |
|---|---|
| `augmentation.color_jitter.brightness` | (float) the maximum scale of adjusting the brightness of an image. The scale value is selected within range. |
| `augmentation.color_jitter.contrast` | (float) the maximum scale of adjusting the contrast of an image. The scale value is selected within range. |
| `augmentation.color_jitter.saturation` | (float) the maximum scale of adjusting the saturation of an image. The scale value is selected within range. |
| `augmentation.color_jitter.hue` | (float) the maximum scale of adjusting the hue of an image. The scale value is selected within range. |
| `augmentation.color_jitter.colorjitter_p` | (float) the probability of applying color jitter. If `1.0`, it always applies the color transform. |