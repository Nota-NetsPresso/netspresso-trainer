# Augmentation - Overview

NetsPresso Trainer provides data augmentation functions to improve model performance, allowing users to configure their own training recipes as desired. 
Data augmentation in NetsPresso Trainer is based on torch and torchvision, and all augmentations are implemented based on `pillow` images.

In NetsPresso Trainer, users can create their desired augmentation recipe by composing a configuration as below. We separately define sample transform procedures for training and inference. 

Functions specified in the `train` and `inference` fields are applied sequentially as listed. Note that after all image processing is completed, the final image size must match with the size specified in `augmentation.img_size`.


```yaml
augmentation:
  img_size: &img_size 256
  train:
    - 
      name: randomresizedcrop
      size: *img_size
      scale: [0.08, 1.0]
      ratio: [0.75, 1.33]
      interpolation: bilinear
    - 
      name: randomhorizontalflip
      p: 0.5
    -
      name: mixing
      mixup: [0.25, 1.0]
      cutmix: ~
      inplace: false
  inference:
    - 
      name: resize
      size: [*img_size, *img_size]
      interpolation: bilinear
      max_size: ~

```

## Gradio demo for simulating the transform

In many learning function repositories, it is recommended to read the code and documentation or actually run the training to check the logs to see how augmentations are performed. 
NetsPresso Trainer supports augmentation simulation to help users easily understand the augmentation recipe they have configured. 
By copying and pasting the augmentation configuration into the simulator, users can preview how a specific image will be augmented in advance. However, transforms (e.g. Normalize, ToTensor) used to convert the image array for learning purposes are excluded from the simulation visualization process. 
In particular, since this simulator directly imports the augmentation modules used in NetsPresso Trainer, users can use the same functions as the augmentation functions used in actual training to verify the results.  

Our team hopes that the learning process with NetsPresso Trainer will become a more enjoyable experience for all users. 

### Running on your environment

Please run the gradio demo with following command:

```bash
bash scripts/run_simulator_augmentation.sh
```

## Field list

| Field <img width=200/> | Description |
|---|---|
| `augmentation.img_size` | (int) The image size of model input after finishing the data augmentation |
| `augmentation.train` | list[dict] List of transform functions for training. Augmentation process is defined on list order. |
| `augmentation.inference` | (list[dict]) List of transform functions for inference. Augmentation process is defined on list order. |

